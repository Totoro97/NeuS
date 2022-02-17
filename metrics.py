import argparse
import hashlib
import os
from typing import Dict, Union, Tuple

import pytorch3d
import torch
import torch.utils.data
import trimesh
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes, Pointclouds

from utils.comm import synchronize, is_main_process, get_world_size
from utils.compute_iou import compute_iou
from utils.inside_mesh import inside_mesh
from utils.point_to_face import point_mesh_face_distances


class MeshEvaluator:
    def __init__(self, path_gt: str, num_samples: int = 100000, device: torch.device = torch.device('cuda')):
        self.path_gt: str = path_gt
        self.num_samples: int = num_samples
        self.device: torch.device = device
        self.mesh_gt: Meshes = self.read_mesh(path_gt)
        self.bbox_gt: torch.Tensor = self.mesh_gt.get_bounding_boxes()  # (1, 3, 2)
        self.samples_gt, self.normals_gt = self.get_gt_samples()

    def get_gt_samples(self) -> Tuple[torch.Tensor, torch.Tensor]:
        cache_dir: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path: str = os.path.join(cache_dir, f"{md5(self.path_gt)}_{self.num_samples}.pt")

        if get_world_size() > 1 and not os.path.exists(cache_path) and not is_main_process():
            synchronize()

        if os.path.exists(cache_path):
            samples_gt, normals_gt = torch.load(cache_path, map_location=self.device).split(1)
        else:
            assert is_main_process()
            samples_gt, normals_gt = sample_points_from_meshes(self.mesh_gt, num_samples=self.num_samples,
                                                               return_normals=True)  # (1, n, 3), (1, n, 3)
            torch.save(torch.cat([self.samples_gt, self.normals_gt], dim=0).cpu(), cache_path)
            if get_world_size() > 1:
                synchronize()

        return samples_gt, normals_gt

    @property
    def max_side(self) -> float:
        if getattr(self, "_max_side", None) is None:
            sides = self.bbox_gt[:, :, 1] - self.bbox_gt[:, :, 0]  # (1, 3)
            self._max_side = sides.max().item()
        return self._max_side

    def read_mesh(self, path: str) -> Meshes:
        mesh = trimesh.load_mesh(path)
        return Meshes(
            verts=[torch.from_numpy(mesh.vertices).float()],
            faces=[torch.from_numpy(mesh.faces).long()]
        ).to(self.device)

    def compute_chamfer(self, samples_pred, normals_pred) -> Dict:
        chamfer_distance, normals_loss = \
            pytorch3d.loss.chamfer_distance(x=samples_pred, y=self.samples_gt,
                                            x_normals=normals_pred, y_normals=self.normals_gt)
        normal_consistency = 1 - normals_loss

        # Like Fan et al. [17] we use 1/10 times the maximal edge length
        # of the current object’s bounding box as unit 1
        unit = self.max_side / 10.0
        unit2 = unit ** 2
        chamfer_distance /= unit2

        return {
            'chamfer-l2': chamfer_distance.item(),
            'normals': normal_consistency.item()
        }

    def compute_iou(self, mesh_pred: Meshes) -> Dict:
        # valid only for watertight meshes
        # todo currently leads to oom when the num_samples is big

        bbox_gt = self.bbox_gt  # (1, 3, 2)
        bbox_pred = mesh_pred.get_bounding_boxes()  # (1, 3, 2)

        bbox_cat = torch.cat([bbox_pred, bbox_gt], dim=0)
        bound_min = bbox_cat.min(dim=0).values[None, :, 0]  # (1, 3)
        bound_max = bbox_cat.max(dim=0).values[None, :, 1]  # (1, 3)

        samples_range = (bound_max - bound_min)
        vol_samples = bound_min + torch.rand(self.num_samples, 3, device=bound_min.device) * samples_range  # (n, 3)
        occ_gt = inside_mesh(vol_samples, self.mesh_gt.verts_list()[0], self.mesh_gt.faces_list()[0])
        occ_pred = inside_mesh(vol_samples, mesh_pred.verts_list()[0], mesh_pred.faces_list()[0])
        iou = compute_iou(occ_pred, occ_gt)

        return {'iou': iou}

    def compute_fscore(self, samples_pred: torch.Tensor, mesh_pred: Meshes, qs=(0.01, 0.02, 0.04)) -> Dict:
        p_dists2 = point_mesh_face_distances(self.mesh_gt, Pointclouds(samples_pred))
        r_dists2 = point_mesh_face_distances(mesh_pred, Pointclouds(self.samples_gt))
        fs = {}
        for q in qs:
            d = q * self.max_side
            d2 = d ** 2
            precision = (p_dists2 <= d2).sum() / p_dists2.shape[0]
            recall = (r_dists2 <= d2).sum() / r_dists2.shape[0]
            fscore = 2.0 * precision * recall / (precision + recall)
            fs[f'f-score-{q}'] = fscore.item()
        return fs

    @torch.no_grad()
    def compute_metrics(self, mesh_pred: Union[str, Meshes]) -> Dict:
        # the volumetric IoU and a normal consistency score
        # are defined in https://arxiv.org/pdf/1812.03828.pdf
        # f-score is defined in https://arxiv.org/pdf/1905.03678.pdf
        # code reference: https://github.com/autonomousvision/occupancy_networks/blob/406f79468fb8b57b3e76816aaa73b1915c53ad22/im2mesh/eval.py

        if isinstance(mesh_pred, str):
            mesh_pred = self.read_mesh(mesh_pred)

        samples_pred, normals_pred = sample_points_from_meshes(mesh_pred, num_samples=self.num_samples,
                                                               return_normals=True)
        metrics = {
            # **self.compute_iou(mesh_pred),
            **self.compute_fscore(samples_pred, mesh_pred),
            **self.compute_chamfer(samples_pred, normals_pred)
        }

        return metrics


def md5(path):
    with open(path, "rb") as f:
        file_hash = hashlib.md5()
        chunk = f.read(8192)
        while chunk:
            file_hash.update(chunk)
            chunk = f.read(8192)
    return file_hash.hexdigest()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh', type=str, default=None)
    parser.add_argument('--gt_mesh', type=str, default=None)
    args = parser.parse_args()

    evaluator = MeshEvaluator(args.gt_mesh)
    metrics = evaluator.compute_metrics(args.mesh)
    print(metrics)
