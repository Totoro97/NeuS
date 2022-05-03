import pdb

import os
import numpy as np
import cv2


def convert_NDC_to_screen(
    im_w, im_h, fx_ndc, fy_ndc, px_ndc, py_ndc
):
    s = min(im_w, im_h)
    px_screen = -(px_ndc * s / 2) + im_w / 2
    py_screen = -(py_ndc * s / 2) + im_h / 2
    fx_screen = fx_ndc * s / 2
    fy_screen = fy_ndc * s / 2
    return fx_screen, fy_screen, px_screen, py_screen


def convert_screen_to_NDC(
        image_width, image_height, fx_screen, fy_screen, px_screen, py_screen
):
    s = min(image_width, image_height)
    fx_ndc = fx_screen * 2.0 / s
    fy_ndc = fy_screen * 2.0 / s

    px_ndc = -(px_screen - image_width / 2.0) * 2.0 / s
    py_ndc = -(py_screen - image_height / 2.0) * 2.0 / s
    return fx_ndc, fy_ndc, px_ndc, py_ndc


def main():
    """Given a directory of CO3D data, converts it into nice NeUS data."""
    # Filtering based on whether "image" exist instead of images
    cases = [n for n in os.listdir(".") if os.path.isdir(n)]
    # The CO3D data has "images" instead of "image"
    cases = [n for n in cases if not os.path.exists(os.path.join(n, "image"))]

    for idx, case in enumerate(cases):
        print(f"Processing {case}: ({idx + 1}/{len(cases)})", end='\r')
        # Renaming stuff
        i_dir = os.path.join(case, "image")
        m_dir = os.path.join(case, "mask")
        os.rename(os.path.join(case, "images"), i_dir)
        os.rename(os.path.join(case, "masks"), m_dir)
        for im_name in [os.path.join(i_dir, n) for n in os.listdir(i_dir)]:
            os.rename(im_name, im_name.replace("frame", ""))
        for im_name in [os.path.join(m_dir, n) for n in os.listdir(m_dir)]:
            os.rename(im_name, im_name.replace("frame", ""))

        # Changing images to png
        for im_name in [os.path.join(i_dir, n) for n in os.listdir(i_dir)]:
            cv2.imwrite(im_name.replace(".jpg", ".png"), cv2.imread(im_name))
            os.remove(im_name)
        # Changing masks to binary
        for im_name in [os.path.join(m_dir, n) for n in os.listdir(m_dir)]:
            img = cv2.imread(im_name)
            cv2.imwrite(im_name, (img > 127).astype(np.uint8) * 255)

        # Camera parameters
        co_param = np.load(f"{case}/params.npz", allow_pickle=True)
        params = co_param['arr_0'].item()['frame_params']
        scale_mat = np.array(co_param['arr_0'].item()['scale'])
        scale_mat_zeros = np.zeros((4, 4), dtype=scale_mat.dtype)
        scale_mat_zeros[3, 3] = 1
        scale_mat_zeros[:3, :3] = scale_mat
        scale_mat = scale_mat_zeros
        neus_param = {}
        for data in params:
            name = data['path'].stem
            name_idx = name.replace("frame", "")

            h, w = data['size']
            R = np.array(data['R']).T
            T = np.array(data['T'])
            ff = np.array(data['focal_length'])
            pp = np.array(data['principal_point'])

            K = np.zeros((3, 3), dtype=R.dtype)
            K[0, 0], K[1, 1], K[0, 2], K[1, 2] = convert_NDC_to_screen(w, h, ff[0], ff[1], pp[0], pp[1])
            K[2, 2] = 1

            P = (K @ np.concatenate((R, T[:, None]), axis=1))
            P = np.concatenate((P, np.zeros((1, 4))), axis=0)
            P[3, 3] = 1
            neus_param[f'world_mat_{name_idx}'] = P
            neus_param[f'scale_mat_{name_idx}'] = scale_mat
        np.savez(f"{case}/cameras_sphere.npz", neus_param)

    for idx, case in enumerate(sorted(cases)):
        print(f"Renamed case {case} to {idx:05d}")
        os.rename(case, f"{idx:05d}")


if __name__ == "__main__":
    main()
