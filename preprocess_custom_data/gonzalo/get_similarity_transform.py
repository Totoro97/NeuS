import numpy as np

# Source: https://zpl.fi/aligning-point-patterns-with-kabsch-umeyama-algorithm/
def get_similarity_transform(A, B):
    """
    Compute the optimal similarity transform from B to A.
    Namely, compute rotation `R`, translation `t` and scale `c` that best align points `B` into
    reference points `A` in the least squares sense as `c * R @ b_i + t ≈ a_i` for all `i`.
    Or, in matrix form,
        (c*R[3,3] | t ) @ (B[3,N]) ≈ (A[3,N])
        (    0    | 1 ) @ (  1   )   (  1   )
    Uses the Kabsch-Umeyama algorithm.

    A, B
        np.ndarray, float32, (N, 3)

    return:
    R
        np.ndarray, float32, (3, 3)
    t
        np.ndarray, float32, (3,)
    c
        float
    """
    assert A.shape == B.shape
    n, m = A.shape

    EA = np.mean(A, axis=0)
    EB = np.mean(B, axis=0)
    VarA = np.mean(np.linalg.norm(A - EA, axis=1) ** 2)

    H = ((A - EA).T @ (B - EB)) / n
    U, D, VT = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U) * np.linalg.det(VT))
    S = np.diag([1] * (m - 1) + [d]).astype(A.dtype)

    R = U @ S @ VT
    c = VarA / np.trace(np.diag(D) @ S)
    t = EA - c * R @ EB

    return R, t, c


if __name__ == '__main__':
    TARGET = np.float32([
        [-0.067039, 0.037248, 0.257207],
        [-0.021285, 0.123967, 0.23051],
        [-0.138212, 0.123967, 0.235361],
        [-0.077207, 0.310112, 0.13561],
        [-0.077207, -0.048883, 0.170014],
    ])

    SOURCE = np.float32([
        [-0.025177, 0.05771, 0.226592],
        [0.055307, 0.167654, 0.207354],
        [-0.111732, 0.160782, 0.215266],
        [-0.034637, 0.449523, 0.08648],
        [-0.021788, -0.052372, 0.10771],
    ])

    R, t, c = get_similarity_transform(TARGET, SOURCE)
    print(f"R:\n{R}")
    print(f"t:\n{t}")
    print(f"scale:\n{c}")
