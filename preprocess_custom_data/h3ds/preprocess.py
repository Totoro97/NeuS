import argparse
import logging
import pathlib
import pickle
import sys
from time import perf_counter
from typing import List

import cv2
import numpy as np

sys.path.append('..')
sys.path.append('../gonzalo')

try:
    import h3ds

    assert h3ds.__version__ == "0.1.1"
    from h3ds.dataset import H3DS
except ModuleNotFoundError:
    pass

from gonzalo.preprocess import ImageCropper, triangulate_anchor_landmarks, \
    get_similarity_transform_to_reference, resave_cameras_npz

logger = logging.getLogger('h3ds')


def preprocess_folders(folders: List[pathlib.Path]):
    """
    Given a list of paths to scenes, preprocess each of them.

    folders:
        list of pathlib.Path
    """
    image_cropper = ImageCropper()
    for i, folder in enumerate(folders):
        logger.info(f"Preprocessing {folder} ({i + 1}/{len(folders)})")
        start_time = perf_counter()
        preprocess_folder(folder, image_cropper)
        end_time = perf_counter()
        logger.info(f"Done, took {end_time - start_time} seconds")


def preprocess_folder(folder: pathlib.Path, image_cropper: ImageCropper, num_views=3):
    h3ds_dir = folder.parent
    scene_id = folder.name
    h3ds = H3DS(path=h3ds_dir)

    output_path = pathlib.Path("/gpfs/gpfs0/3ddl/datasets/H3DS_processed") / scene_id
    output_images_preview_path = output_path / "images-preview"
    output_images_path = output_path / "image"
    output_masks_path = output_path / "mask"
    cameras_sphere_npz_path = output_path / "cameras_sphere.npz"
    output_path.mkdir(exist_ok=True)
    output_images_preview_path.mkdir(exist_ok=True)
    output_images_path.mkdir(exist_ok=True)
    output_masks_path.mkdir(exist_ok=True)

    views_config_id = str(num_views)
    views_idx = h3ds.helper._config['scenes'][scene_id]['default_views_configs'][views_config_id]
    labels = ['{0:04}'.format(idx) for idx in views_idx]
    mesh, _, _, cameras = h3ds.load_scene(scene_id=scene_id, views_config_id=views_config_id)

    camera_matrices = []  # N, 3, 4
    human_data = {
        'crop_rectangles': [],
        'landmarks': [],
    }

    for label, (intrinsics, pose) in sorted(zip(labels, cameras), key=lambda x: x[0]):
        image_path = folder / "image" / f"img_{label}.jpg"
        mask_path = folder / "rigid_masks" / f"mask_{label}.jpg"
        image = cv2.imread(str(image_path))

        try:
            crop_rectangle, _, landmarks = image_cropper.crop_to_face(image)
        except ImageCropper.BlinkException:
            logger.info(f"Blink in frame {image_path}, skipping")
            continue
        except ImageCropper.NoFaceException:
            logger.info(f"No face detected in frame {image_path}, skipping")
            continue

        human_data['crop_rectangles'].append(crop_rectangle)
        human_data['landmarks'].append(landmarks)

        world_mat = np.eye(4)
        world_mat[:3] = intrinsics @ np.linalg.inv(pose)[:3]
        camera_matrices.append(world_mat[:3])

        cv2.imwrite(str(output_images_path / f"{label}.png"), image)
        mask = cv2.imread(str(mask_path))
        cv2.imwrite(str(output_masks_path / f"{label}.png"), mask)

        l, t, r, b, _ = map(int, crop_rectangle)
        cv2.rectangle(image, (l, t), (r, b), (0, 0, 255), 4)
        cv2.imwrite(str(output_images_preview_path / f"{label}.png"), cv2.resize(image, None, fx=1 / 4, fy=1 / 4))

    # Save crop parameters and landmarks, just in case
    with open(output_path / "tabular_data.pkl", 'wb') as f:
        pickle.dump(human_data, f)

    # Estimate several landmarks' 3D coordinates for registration
    logger.info(f"Triangulating anchor landmarks to compute the similarity transform")

    landmarks_2d = human_data['landmarks']  # (the 3rd coordinate is dropped later)
    landmarks_3d = triangulate_anchor_landmarks(camera_matrices, landmarks_2d)  # 6, 3

    # Get transform that roughly brings current 3D landmarks to reference scene's landmarks
    # (all in the coordinate systems of COLMAP's output)
    registration_transform = get_similarity_transform_to_reference(landmarks_3d)  # 4, 4
    logger.info(f"Estimated similarity transform towards reference: {registration_transform.tolist()}")

    # After running COLMAP, NeuS' routine (`neus_gen_cameras` above) computes a transform that
    # brings the object into 1-sphere (from the cleaned point cloud's bounding box) and saves
    # it to "cameras_sphere.npz" as 'scale_mat_XX' (same for all XX).
    # We compute it only ONCE for the "reference scene".
    # For all other scenes, we first realign them using the above `registration_transform`,
    # and then reuse 'scale_mat_XX' from the reference scene.
    resave_cameras_npz(camera_matrices, registration_transform, cameras_sphere_npz_path)

    # todo resave gt mesh
    # logger.info(f"Saving gt mesh in reference coordinates: {registration_transform.tolist()}")


if __name__ == '__main__':
    # python preprocess.py --folders f1 f2 f3 f4 f5
    parser = argparse.ArgumentParser()
    parser.add_argument('folders', type=pathlib.Path, nargs='+')
    args = parser.parse_args()
    preprocess_folders(args.folders)
