""" M4N9 Computational Linear Algebra - Project 2
Tudor Trita Trita
CID: 01199397

Implemented using Python 3.7 (need f-string python 3.6+)

Question 1. Conditioning and Stability for QR and SVD
"""
import numpy as np


def householder_stability(N=20, display=False):
    """Computes norms for QR factorisation"""
    B = np.random.randn(N, N)  # i.i.d normal matrix
    C = np.random.randn(N, N)  # i.i.d normal matrix
    Q, _ = np.linalg.qr(B)  # Forming orthogonal Matrix
    R = np.triu(C)  # Casting C into upper-triangular form

    # Forming A with Q & R as 'exact QR factorisation'
    # Note: minimal rounding applies but is negligible here
    A = Q @ R
    Q2, R2 = np.linalg.qr(A)  # Numerical householder QR
    norm1 = np.linalg.norm(Q2 - Q)
    norm2 = np.linalg.norm(R2 - R)
    norm3 = np.linalg.norm(Q2 @ R2 - A)

    if display:
        print("||Q2 - Q|| = %s" %norm1)
        print("||R2 - R|| = %s" %norm2)
        print("||Q2R2 - A|| = %s" %norm3)
    return (norm1, norm2, norm3)


def svd_stability(N=20, display=False):
    """Computes norms for SVD"""
    B = np.random.randn(N, N)  # i.i.d normal matrix
    C = np.random.randn(N, N)  # i.i.d normal matrix
    S = np.sort(np.abs(np.random.randn(N)))[::-1]  # i.i.d normal vector

    # Generating U, V, sigma using QR (Cheap way to get orthogonal matrices)
    U, _ = np.linalg.qr(B)  # U
    V, _ = np.linalg.qr(C)  # V

    A = (U * S) @ V.T  # Forming A

    U2, S2, VT2 = np.linalg.svd(A)

    norm1 = np.linalg.norm(U2 - U)
    norm2 = np.linalg.norm(S2 - S)
    norm3 = np.linalg.norm(VT2 - V.T)
    norm4 = np.linalg.norm(((U2 * S2) @ VT2) - A)

    if display:
        print("||U2 - U|| = %s" %norm1)
        print("||sigma2 - sigma|| = %s" %norm2)
        print("||V2 - V|| = %s" %norm3)
        print("||(U2 @ S2 @ V2) - A|| = %s" %norm4)
    return (norm1, norm2, norm3, norm4)


def main():
    print("MACHINE EPSILON FOR DOUBLE PRECISION:")
    print(np.finfo(float).eps)
    print()
    qr_iterations = 5000
    print("Performing stability analysis for Householder " \
          "(%s iterations):" %qr_iterations)
    norms_array_householder = np.zeros((qr_iterations, 3))
    for n1 in norms_array_householder:
        n1[:] = householder_stability()

    # Taking averages:
    ave_norms_array_householder = norms_array_householder.mean(axis=0)
    print("Averages for Householder Norms over %s iterations:" %qr_iterations)
    print("||Q2 - Q|| = %s" %ave_norms_array_householder[0])
    print("||R2 - R|| = %s" %ave_norms_array_householder[1])
    print("||Q2R2 - A|| = %s" %ave_norms_array_householder[2])
    print()

    svd_iterations = 5000
    print("Performing stability analysis for SVD " \
          "(%s iterations):" %svd_iterations)
    norms_array_svd = np.zeros((svd_iterations, 4))
    for n2 in norms_array_svd:
        n2[:] = svd_stability()

    # Taking averages:
    ave_norms_array_svd = norms_array_svd.mean(axis=0)
    print("Averages for SVD Norms over %s iterations:" %svd_iterations)
    print("||U2 - U|| = %s" %ave_norms_array_svd[0])
    print("||sigma2 - sigma|| = %s" %ave_norms_array_svd[1])
    print("||VH2 - VH|| = %s" %ave_norms_array_svd[2])
    print("||(U2 @ S2 @ V2) - A|| = %s" %ave_norms_array_svd[3])
    return


if __name__ == "__main__":
    print("Program Started")
    print()
    main()
    print()
    print("Program Finished")
