"""
Preprocess videos from the "Gonzalo's dataset" for learning NeRF-like 3D portraits.

1. В калиброванных timestamp'ах кадров ищем два соседних со вспышкой по её timestamp'у и выкидываем их.
2. Выкидываем первые 40%.
3. Выкидываем кадры, где OpenCV не нашёл открытых глаз.
4. Равномерно выкидываем кадры так, чтобы осталось ~65.
5. COLMAP.
6. Находим лицо, увеличиваем кроп в 2 раза, запоминаем параметры кропа.
7. Обрезаем картинки по кропу.
8. Считаем сегментацию и сохраняем в mask/
9. В коде загрузки датасета NeuS подгоняем интринсики каждого кадра согласно кропу.
9.1. Для этого модифицируем код обучения так, чтобы он использовал не одни интринсики для всех кадров, а разные.
9.2. И ещё пишем загрузчик данных, который загружает данные только когда нужно и не переполняется (проверить, сколько надо памяти).
10. Находим ~6 2D-лэндмарков на каждом кадре.
11. Триангулируем. Запоминаем. Можно туда же, где и кроп.
12. Находим преобразование (регистрируем) к референсной сцене.
"""
from get_similarity_transform import get_similarity_transform

import argparse
import pathlib
import logging
import time
import csv
import bisect
import sys
import subprocess
import pickle
import shutil

import torch
import numpy as np
import cv2

try:
    import face_alignment
    from face_alignment.detection.sfd import FaceDetector
except ImportError:
    raise ImportError(
        "Please install face alignment package from "
        "https://github.com/1adrianb/face-alignment")

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")


class ImageCropper:
    """
    Checks if the face in the image has an open eye and detects landmarks.
    """
    class BlinkException(Exception):
        pass

    def __init__(self):
        self.landmark_detector = face_alignment.FaceAlignment(
            face_alignment.LandmarksType._3D, flip_input=False)

    @staticmethod
    def choose_one_detection(frame_faces):
        """
        frame_faces
            list of lists of length 5
            several face detections from one image

        return:
            list of 5 floats
            one of the input detections: `(l, t, r, b, confidence)`
        """
        # Filter by confidence
        frame_faces = [x for x in frame_faces if x[-1] > 0.87]

        if len(frame_faces) == 0:
            raise ImageCropper.BlinkException
        elif len(frame_faces) == 1:
            return 0
        else:
            # sort by area, find the largest box
            largest_area, largest_idx = -1, -1
            for idx, face in enumerate(frame_faces):
                area = abs(face[2]-face[0]) * abs(face[1]-face[3])
                if area > largest_area:
                    largest_area = area
                    largest_idx = idx

            return largest_idx

    def crop_to_face(self, image):
        """
        image
            np.ndarray, uint8, H x W x 3, BGR

        return:
        crop_rectangle
            list[5] of (int, int, int, int, float)
            Crop rectangle relative to the original image: (l, t, r, b, confidence).
        cropped_image
            np.ndarray, uint8, H1 x W1 x 3, BGR
            `image` cropped according to `crop_rectangle`.
        landmarks
            list[68] of lists[3] of float
            3D facial landmarks' coordinates.
        """
        def enlarge_crop(rect, factor_w, factor_h):
            l, t, r, b, confidence = rect

            center_x = (l + r) / 2
            center_y = (t + b) / 2
            w = r - l
            h = b - t

            return [
                center_x - w / 2 * factor_w,
                center_y - h / 2 * factor_h,
                center_x + w / 2 * factor_w,
                center_y + h / 2 * factor_h,
                confidence]

        def crop(image, rect):
            l, t, r, b, confidence = rect

            l = round(max(l, 0))
            t = round(max(t, 0))
            r = round(max(r, l + 1))
            b = round(max(b, t + 1))
            return image[t:b, l:r], [l, t, r, b, confidence]

        landmarks_list, _, face_rects = self.landmark_detector.get_landmarks_from_image(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB), return_bboxes=True)

        best_face_rect_idx = self.choose_one_detection(face_rects)
        landmarks = landmarks_list[best_face_rect_idx].tolist()
        face_rect = face_rects[best_face_rect_idx].tolist()

        ENLARGE_FACTOR_W = 3.5
        ENLARGE_FACTOR_H = 2.3

        crop_rect = enlarge_crop(face_rect, ENLARGE_FACTOR_W, ENLARGE_FACTOR_H)
        image_cropped, crop_rect_rounded = crop(image, crop_rect)

        return crop_rect_rounded, image_cropped, landmarks


def run_subprocess(command, working_dir=None):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, cwd=working_dir)
    for c in iter(lambda: process.stdout.read(1), b''):
        sys.stdout.buffer.write(c)


def preprocess_folders(folders: list[pathlib.Path]):
    """
    Given a list of paths to scenes, preprocess each of them.
    One scene is a folder, e.g. '132/2021-07-22-12-24-33', which has the 'smartphone_video_frames'
    folder inside following the original dataset structure.

    folders:
        list of pathlib.Path
    """
    logger = logging.getLogger('preprocess_folders')

    image_cropper = ImageCropper()

    this_dir = pathlib.Path(__file__).parent
    if not (this_dir / "Graphonomy/data/model/universal_trained.pth").is_file():
        logger.critical(
            "Please get the background segmentation system " \
            "(`git submodule init --update` and download checkpoints)")
        raise FileNotFoundError

    start_time = time.time()
    for i, folder in enumerate(folders):
        logger.info(f"Preprocessing {folder} ({i+1}/{len(folders)})")

        preprocess_folder(folder, image_cropper)

        end_time = time.time()
        logger.info(f"Done, took {end_time - start_time} seconds")
        start_time = end_time


def preprocess_folder(folder: pathlib.Path, image_cropper: ImageCropper):
    logger = logging.getLogger('preprocess_folder')

    video_data_path = folder / "smartphone_video_frames"

    all_timestamps, flash_timestamps, video_path = get_timestamps_and_video(video_data_path)
    assert sorted(all_timestamps) == all_timestamps, f"{all_timestamps_csv_path} isn't sorted"

    frames_to_keep = select_few_frames(all_timestamps, flash_timestamps)
    logger.info(f"Will save {sum(frames_to_keep)} frames from video {folder}")

    # Create output directories
    output_path = folder / "portrait_reconstruction"
    output_path.mkdir(exist_ok=True)

    images_output_path = output_path / "images-original"
    images_output_path.mkdir(exist_ok=True)
    images_preview_path = output_path / "images-preview"
    images_preview_path.mkdir(exist_ok=True)
    undistorted_images_path = output_path / "images-undistorted"

    # Decode video. Filter and resave frames to a temporary folder
    decode_video(video_path, images_output_path, images_preview_path, frames_to_keep)

    # Clean the dataset (manually for now)
    logger.info(
        f"Now please manually remove blinks from 'images-original' (smaller size in " \
        f"'images-preview' for your convenience) and press Enter")
    input()

    # Run COLMAP
    logger.info(f"Running COLMAP")

    # LLFF's imgs2poses needs 'images'
    symlink_for_colmap = output_path / "images"
    symlink_for_colmap.symlink_to(images_output_path.name, target_is_directory=True)

    sys.path.insert(1, "../colmap_preprocess")
    from pose_utils import gen_poses as llff_gen_poses
    # Generates 'poses.npy', 'sparse/0', 'sparse_points.ply'
    llff_gen_poses(output_path, 'exhaustive_matcher')

    IS_REFERENCE_SCENE = False
    if IS_REFERENCE_SCENE:
        logger.info(
            f"Отлично мужик, а теперь open 'sparse_points.ply' with MeshLab, clean the point " \
            f"cloud and resave it as 'sparse_points_interest.ply', then press Enter")
        input()
    else:
        shutil.copy(
            output_path / "sparse_points.ply",
            output_path / "sparse_points_interest.ply")

    # Convert camera poses from LLFF to IDR (= NeuS) format (will generate 'cameras_sphere.npz')
    from gen_cameras import process_dir as neus_gen_cameras
    neus_gen_cameras(output_path, resave_images=False)

    # Undistort images
    run_subprocess([
        "colmap", 'image_undistorter',
        '--image_path', "./images-original/",
        '--input_path', "./sparse/0/",
        '--output_path', "./dense/",
        '--output_type', 'COLMAP',
        '--min_scale', "1",
        '--max_scale', "1"],
        working_dir=output_path)

    symlink_for_colmap.unlink()

    # The undistorted images are the final ones to be used during training, move them
    (output_path / "dense/images").rename(undistorted_images_path)
    (output_path / "image").symlink_to(undistorted_images_path.name, target_is_directory=True)

    # Detect faces and landmarks
    logger.info(f"Detecting face and landmarks, visualizing at {images_preview_path}")

    human_data = {
        'crop_rectangles': [],
        'landmarks': [],
    }

    for image_path in sorted(undistorted_images_path.iterdir()):
        image = cv2.imread(str(image_path))

        try:
            crop_rectangle, _, landmarks = image_cropper.crop_to_face(image)
        except ImageCropper.BlinkException:
            logger.info(f"Blink in frame {frame_idx}, skipping")
            continue

        human_data['crop_rectangles'].append(crop_rectangle)
        human_data['landmarks'].append(landmarks)

        l, t, r, b, _ = map(int, crop_rectangle)
        cv2.rectangle(image, (l, t), (r, b), (0, 0, 255), 4)
        cv2.imwrite(
            str(images_preview_path / image_path.name), cv2.resize(image, None, fx=1/4, fy=1/4))

    # Save crop parameters and landmarks, just in case
    with open(output_path / "tabular_data.pkl", 'wb') as f:
        pickle.dump(human_data, f)

    # Compute background masks
    logger.info(f"Predicting background masks")

    masks_output_path = output_path / "mask"
    masks_output_path.mkdir(exist_ok=True)

    working_dir = "./Graphonomy"
    command = [
        "python3", '-u', "exp/inference/inference_folder.py",
        '--images_path', f"{undistorted_images_path.resolve()}",
        '--output_dir', f"{masks_output_path.resolve()}",
        '--model_path', "data/model/universal_trained.pth",
        '--tta', "0.75,1.0,1.2,1.4",]
    run_subprocess(command, working_dir)

    # Estimate several landmarks' 3D coordinates for registration
    logger.info(f"Triangulating anchor landmarks to compute the similarity transform")

    cameras_sphere_npz_path = output_path / "cameras_sphere.npz"
    camera_matrices_file = np.load(cameras_sphere_npz_path)
    num_images = len(list(undistorted_images_path.iterdir()))
    camera_matrices = \
        [camera_matrices_file[f'world_mat_{i}'][:3] for i in range(num_images)] # N, 3, 4

    with open(output_path / "tabular_data.pkl", 'rb') as f:
        landmarks_2d = pickle.load(f)['landmarks'] # N, 68, 3 (the 3rd coordinate is dropped later)

    landmarks_3d = triangulate_anchor_landmarks(camera_matrices, landmarks_2d) # 6, 3

    # Get transform that roughly brings current 3D landmarks to reference scene's landmarks
    # (all in the coordinate systems of COLMAP's output)
    registration_transform = get_similarity_transform_to_reference(landmarks_3d) # 4, 4
    logger.info(f"Estimated similarity transform towards reference: {registration_transform.tolist()}")

    # After running COLMAP, NeuS' routine (`neus_gen_cameras` above) computes a transform that
    # brings the object into 1-sphere (from the cleaned point cloud's bounding box) and saves
    # it to "cameras_sphere.npz" as 'scale_mat_XX' (same for all XX).
    # We compute it only ONCE for the "reference scene".
    # For all other scenes, we first realign them using the above `registration_transform`,
    # and then reuse 'scale_mat_XX' from the reference scene.
    resave_cameras_npz(camera_matrices, registration_transform, cameras_sphere_npz_path)

    logger.info(f"Cleaning up")
    # COLMAP-related
    shutil.rmtree(output_path / "dense")
    shutil.rmtree(output_path / "sparse")
    (output_path / "database.db").unlink()
    # NeuS' routine-related
    (output_path / "poses.npy").unlink()
    (output_path / "sparse_points_interest.ply").unlink()
    # Everything else
    shutil.rmtree(images_output_path) # non-undistorted images


def get_timestamps_and_video(folder):
    files_in_dir = list(folder.iterdir())

    def find_file_with_suffix(paths, suffix):
        try:
            return next(x for x in paths if x.name.endswith(suffix))
        except StopIteration:
            raise FileNotFoundError(f"No *{suffix} in {paths}")

    all_timestamps_csv_path = find_file_with_suffix(files_in_dir, '_aligned_timestamps.csv')
    flash_timestamps_csv_path = find_file_with_suffix(files_in_dir, '_aligned_flash.csv')
    video_path = find_file_with_suffix(files_in_dir, '.mp4')

    # Read timestamps from csv
    def read_timestamps(csv_path):
        with open(csv_path, 'r', newline='') as csv_file:
            return [int(row[0]) for row in csv.reader(csv_file)]

    all_timestamps = read_timestamps(all_timestamps_csv_path)
    flash_timestamps = read_timestamps(flash_timestamps_csv_path)

    return all_timestamps, flash_timestamps, video_path


def select_few_frames(
    all_timestamps, flash_timestamps, first_frames_fraction_to_skip=0.4, final_num_frames=75):
    frames_to_keep = [True] * len(all_timestamps)

    # Determine which frames are with flash, remove them

    # `flash_timestamps` tell when the flash is triggered (in nanoseconds).
    # The manually measured rough conservative values below tell what can be the delay between
    # the trigger and the start/stop of the flashlight.
    FLASH_MIN_START_DELAY = 190_000_000
    FLASH_MAX_END_DELAY = 420_000_000

    for flash_timestamp in flash_timestamps:
        flash_first_frame_idx = bisect.bisect(
            all_timestamps, flash_timestamp + FLASH_MIN_START_DELAY)
        flash_last_frame_idx = bisect.bisect(
            all_timestamps, flash_timestamp + FLASH_MAX_END_DELAY)

        # Mark these frames as having flash
        for frame_idx in range(flash_first_frame_idx, flash_last_frame_idx):
            frames_to_keep[frame_idx] = False

    # Skip the first `first_n_frames_to_skip` (e.g. 40%)
    first_n_frames_to_skip = int(len(all_timestamps) * first_frames_fraction_to_skip)
    for i in range(first_n_frames_to_skip):
        frames_to_keep[i] = False

    # Leave only about `final_num_frames` (e.g. 75) frames uniformly
    frames_remaining = sum(frames_to_keep)
    assert frames_remaining > final_num_frames, \
        f"Too few video frames ({frames_remaining} but need "
        f"{sum(frames_to_keep)}), please debug this"

    surviving_frame_idx = 0 # counts how many frames with `frames_to_keep[i] == True` we met
    for i in range(first_n_frames_to_skip, len(all_timestamps)):
        if frames_to_keep[i] is True:
            # Decide to leave it or throw away
            if surviving_frame_idx % (frames_remaining // final_num_frames) != 0:
                frames_to_keep[i] = False

            surviving_frame_idx += 1

    return frames_to_keep


def decode_video(video_path, images_output_path, images_preview_path, frames_to_keep):
    video_reader = cv2.VideoCapture(str(video_path))

    for frame_idx in range(len(frames_to_keep)):
        frame_grab_success = video_reader.grab()
        if frames_to_keep[frame_idx] is False:
            continue

        # if cropped_images_output_path.is_file():
        #     logger.info(f"Frame {frame_idx}/{len(frames_to_keep)} already on disk, skipping")
        #     continue

        frame_retrieve_success, image = video_reader.retrieve()
        assert frame_grab_success and frame_retrieve_success, \
            f"{video_path} exhausted at frame {frame_idx}, expected {len(frames_to_keep)}"

        image_output_path = images_output_path / f"{frame_idx:05d}.png"
        cv2.imwrite(str(image_output_path), image)

        if images_preview_path is not None:
            image_preview_path = images_preview_path / f"{frame_idx:05d}.jpg"
            cv2.imwrite(str(image_preview_path), cv2.resize(image, dsize=None, fx=1/4, fy=1/4))


def triangulate_anchor_landmarks(camera_matrices, landmarks_2d):
    """
    Extract (out of 68) and triangulate these 2D landmarks:
    8 - chin
    33 - nose tip
    37 - ~right eye [36:42]
    43 - ~left eye [42:48]
    0 - right "ear"
    16 - left "ear"
    """
    logger = logging.getLogger('triangulate_anchor_landmarks')

    ANCHOR_LANDMARK_IDXS = [8, 33, 37, 43, 0, 16]

    landmarks_2d = np.array(landmarks_2d)      # N, 68, 3
    landmarks_2d = landmarks_2d[:, ANCHOR_LANDMARK_IDXS, :2] # N, 6, 2

    # Triangulate 10 times (using a different pair each time), then median-average
    camera_indices = \
        [(i*2, len(camera_matrices) // 2 + i*2) for i in range(10)] # deterministic shuffle :)
    landmarks_3d_guesses = [] # eventually 10, 6, 3

    for cam1_idx, cam2_idx in camera_indices:
        landmarks_3d_guess = cv2.triangulatePoints(               # 4, 6
            camera_matrices[cam1_idx], camera_matrices[cam2_idx], # 3, 4
            landmarks_2d[cam1_idx].T, landmarks_2d[cam2_idx].T)   # 2, 6
        landmarks_3d_guess = cv2.convertPointsFromHomogeneous(landmarks_3d_guess.T) # 6, 1, 3

        landmarks_3d_guesses.append(landmarks_3d_guess.squeeze(1)) # 6, 3

    landmarks_3d = np.median(landmarks_3d_guesses, 0) # 6, 3

    logger.info(f"Anchor landmarks' coordinates in the coordinate system of COLMAP's output:")
    for landmark_idx68, landmark_3d in zip(ANCHOR_LANDMARK_IDXS, landmarks_3d):
        logger.info(f"Landmark №{landmark_idx68:02d}: {landmark_3d.tolist()}")

    return landmarks_3d


def get_similarity_transform_to_reference(landmarks_3d):
    """
    landmarks_3d
        np.ndarray, float32, (6, 3)

    return:
    transform
        np.ndarray, float32, (3, 4)
    """
    REFERENCE_LANDMARKS_3D = np.float32([
        [-2.0822412261220538, 1.697800522724807, 4.520114276952494],
        [-1.9801963763579469, 1.2931693187915096, 4.230872368967606],
        [-2.276916147971233, 0.930831168211703, 4.1528538228895435],
        [-1.8265286016971878, 0.8983242853496866, 4.356613008380495],
        [-2.791900772306086, 0.8693908260150472, 4.539910653815297],
        [-1.758480803201661, 0.7613888777243099, 5.032362007722343],
    ])

    # (c*R[3,3] | t ) @ (B[3,N]) ≈ (A[3,N])
    # (    0    | 1 ) @ (  1   )   (  1   )
    # а нам надо такое преобразование S, чтоб
    # S^-1 @ landmarks_3d ~ REFERENCE_LANDMARKS_3D
    # S @ REFERENCE_LANDMARKS_3D ~ landmarks_3d
    R, t, c = get_similarity_transform(landmarks_3d, REFERENCE_LANDMARKS_3D)

    retval = np.zeros((4, 4), REFERENCE_LANDMARKS_3D.dtype)
    retval[:3, :3] = c * R
    retval[:3, 3] = t
    retval[3, 3] = 1.0
    return retval


def resave_cameras_npz(
    camera_matrices, registration_transform, cameras_sphere_npz_path):
    REFERENCE_SCALE_TRANSFORM = np.float32([
        [ 2.5201557,  0.       ,  0.       , -2.3363597],
        [ 0.       ,  2.5201557,  0.       ,  1.6211548],
        [ 0.       ,  0.       ,  2.5201557,  5.1853   ],
        [ 0.       ,  0.       ,  0.       ,  1.       ],])

    dict_to_save = {}
    for camera_idx, camera_matrix in enumerate(camera_matrices):
        dict_to_save[f'world_mat_{camera_idx}'] = camera_matrix @ registration_transform
        dict_to_save[f'scale_mat_{camera_idx}'] = REFERENCE_SCALE_TRANSFORM

    np.savez(cameras_sphere_npz_path, **dict_to_save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folders', type=pathlib.Path, nargs='+')
    args = parser.parse_args()

    preprocess_folders(args.folders)
