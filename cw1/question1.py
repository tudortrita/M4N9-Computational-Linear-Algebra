""" M4N9 Computational Linear Algebra - Project 1
Tudor Trita Trita
CID: 01199397

Implemented using Python 3.7

Question 1. QR Factorisation by Householder Reflection

Notes: This script performs contains the function which performs the QR
       factorisation of a matrix A using Householder reflections.

       The script also performs tests that verify that the function computes
       the correct QR factorisation.

       The tests may be called separately, and from another file as well.
"""
import numpy as np


def householder(A):
    """Performs QR Factorisation using Householder Reflections.
    """
    try:
        N, M = A.shape
        iterations = min(N, M)
        matrix_type = A[0, 0].dtype
    except:
        print("INPUT A MATRIX!")
        return

    if matrix_type == np.float or matrix_type == np.int:
        # Working in real double-precision
        I = np.eye(N, dtype=np.float64)
        Q = np.eye(N, dtype=np.float64)
        R = np.asarray(A.copy(), dtype=np.float64)

        for k in range(iterations):
            x = R[k:, k]  # Assigning kth column of R from kth entry onwards
            e = I[k:, k]  # Assigning kth column of the Identity from kth entry onwards
            sign_x = np.sign(x[0])  # Checking sign of the first entry in x
            sign_x = sign_x if sign_x != 0 else 1  # Setting sign = 1 if sign_x = 0
            v = sign_x*np.sqrt(np.dot(x, x))*e + x  # Calculating v
            v = v / np.sqrt(np.dot(v, v))  # Normalizing v
            R[k:, k:] -= 2 * np.outer(v, v @ R[k:, k:])  # Calculating R in-place
            Q[:, k:] -= 2 * np.dot(Q[:, k:], np.outer(v, v))  # Calculating Q in-place

    elif matrix_type == np.complex:
        # Working in double precision, both real and imaginary are float64 numbers
        I = np.eye(N, dtype=np.complex128)
        Q = np.eye(N, dtype=np.complex128)
        R = np.asarray(A.copy(), dtype=np.complex128)

        for k in range(iterations):
            x = R[k:, k]  # Assigning kth column of R from kth entry onwards
            norm_x = np.sqrt(np.dot(x.conj(), x))  # Computing Norm of X
            sign_x = np.sign(np.real(x[0]))  # Calculating the sign of the real part of the first entry of x
            sign_x = sign_x if sign_x != 0 else 1  # Setting sign = 1 if sign_x = 0
            u_1 = sign_x*norm_x + x[0]  # Computing first value of vector
            v = R[k:, k]/u_1  # Calculating v
            v[0] = (1 + 0j)  # Setting first elemnent of v
            t = sign_x*u_1/norm_x  # Reflection parameters
            R[k:, :] = R[k:, :] - np.outer((t.conj()*v), np.dot(v.conj(), R[k:, :]))  # Calculating R in-place
            Q[:, k:] = Q[:, k:] - np.outer(Q[:, k:] @ v, (t.conj() * v).conj())  # Calculating Q in-place

    else:
        print("Matrix data-type not understood. Please input a nicer matrix!")
    return Q, R


def remove_small_entries(matrix, accuracy=10**-15):
    """ Function which sets all entries < accuracy to zero.
    """
    matrix[np.where(np.abs(matrix) < accuracy)] = 0
    return matrix


def determinant(A):
    """ The following function has been adapted from the website:
    https://integratedmlai.com/find-the-determinant-of-a-matrix-with-pure-python-without-numpy-or-scipy/
    This is the reference to it.
    """
    # Section 1: Establish n parameter and copy A
    n = len(A)
    AM = A.copy()

    # Section 2: Row ops on A to get in upper triangle form
    for fd in range(n): # A) fd stands for focus diagonal
        for i in range(fd+1,n): # B) only use rows below fd row
            if AM[fd][fd] == 0: # C) if diagonal is zero ...
                AM[fd][fd] == 1.0e-18 # change to ~zero
            # D) cr stands for "current row"
            crScaler = AM[i][fd] / AM[fd][fd]
            # E) cr - crScaler * fdRow, one element at a time
            for j in range(n):
                AM[i][j] = AM[i][j] - crScaler * AM[fd][j]

    # Section 3: Once AM is in upper triangle form ...
    product = 1.0
    for i in range(n):
        # ... product of diagonals is determinant
        product *= AM[i][i]
    return product


def check_qr(A, Q, R, display=True, tolerance=10**-12):
    """ Checks errors in the matrix A - QR.
    """
    M = A - Q@R
    largest_error = np.max(M)
    sum_abs_errors = np.abs(M).sum()

    print()
    print("Test 1: Difference 'A - QR':")
    if display:
        print()
        print(M)
        print()
    print(" ## Largest error = %s ##" %(largest_error))
    print(" ## Sum of |errors| = %s ##" %(sum_abs_errors))

    print()
    print(" ## Largest |error| tolerance met? ##")
    print(bool(np.abs(largest_error) < tolerance))
    print()
    return


def check_unitary(Q, display=True, tolerance=10**-12):
    """ Checks that Q is unitary by computing Q*Q and QQ*.

    Notes: np.allclose outputs True if the inputs are within tolerance
           of each other element-wise.
    """
    M1 = Q.conjugate().T @ Q
    M2 = Q @ Q.conjugate().T

    print()
    print("Test 2: Checking that Q is unitary")
    print(" ## Q*Q == I? ##")
    print(np.allclose(M1, np.eye(M1.shape[0]), rtol=tolerance))
    if display:
        M1 = remove_small_entries(M1, accuracy=tolerance)
        print()
        print(M1)
        print()
    print(" ## QQ* = I? ##")
    print(np.allclose(M2, np.eye(M2.shape[0]), rtol=tolerance))
    if display:
        M2 = remove_small_entries(M2, accuracy=tolerance)
        print()
        print(M2)
        print()
    print()
    return


def check_upper_triangular(R, display=True, tolerance=10**-12):
    """ Checks if the matrix is upper triangular within a tolerance level.

    Notes: np.triu sets everything below the diagonal equal to zero.
    """
    print()
    print("Test 3: Checking that R is upper triangular")
    print(" ## R == upper triangular? ##")
    print(np.allclose(R, np.triu(R), rtol=tolerance))
    if display:
        print()
        print(M)
        print()
    print()
    return


def check_determinants(A, Q, R, tolerance=10**-12):
    """ Check determinants of A, Q, R and display the results.
    """
    determinant_matrix_A = np.abs(determinant(A))
    determinant_matrix_R = np.abs(determinant(R))
    determinant_matrix_Q = np.abs(determinant(Q))
    determinant_matrix_QR = np.abs(determinant_matrix_Q*determinant_matrix_R)
    determinant_difference = np.abs(determinant_matrix_A - determinant_matrix_QR)
    print()
    print("Test 4: Checking determinants")
    print("|det(A)| = %s" %(determinant_matrix_A))
    print("|det(R)| = %s" %(determinant_matrix_R))
    print("|det(Q)| = %s" %(determinant_matrix_Q))
    print("|det(QR)| = %s" %(determinant_matrix_QR))
    print("|det(A) - det(QR)| = %s" %(determinant_difference))
    print()
    print(" ## Tolerance met? ##")
    print(bool(determinant_difference < tolerance))
    print()
    return


def perform_tests(A, Q, R, display=False, tolerance=10**-12):
    """ Function to perform all the tests on A, Q and R.
        Toggle display=True to show the resulting matrices in the output.
    """
    print()
    print("Size of A = (%s x %s)" %(A.shape[0], A.shape[1]))
    check_qr(A, Q, R, display=display, tolerance=tolerance)
    check_unitary(Q, display=display, tolerance=tolerance)
    check_upper_triangular(R, display=display, tolerance=tolerance)
    if A.shape[0] == A.shape[1]:
        check_determinants(A, Q, R, tolerance=tolerance)
    return


def main(display=False, tolerance=10**-12, rnd_seed=42):
    """ Main function in the program.
    """
    print()
    print("RUNNING TESTS NOW")
    print()

    np.random.seed(rnd_seed)

    print("--------------------------")
    print("CASE 1: Taking a random REAL square matrix")
    A = np.random.rand(4, 4)
    Q, R = householder(A)
    perform_tests(A, Q, R, display=display, tolerance=tolerance)

    print("--------------------------")
    print("CASE 2: Taking a random COMPLEX square matrix")
    MAT = np.random.rand(4, 4, 2)
    A = MAT[:, :, 0] + 1j*MAT[:, :, 1]
    Q, R = householder(A)
    perform_tests(A, Q, R, display=display, tolerance=tolerance)

    print("--------------------------")
    print("CASE 3: Taking a random COMPLEX rectangular matrix (N > M)")
    MAT = np.random.rand(5, 3, 2)
    A = MAT[:, :, 0] + 1j*MAT[:, :, 1]
    Q, R = householder(A)
    perform_tests(A, Q, R, display=display, tolerance=tolerance)

    print("--------------------------")
    print("CASE 4: Taking a random COMPLEX rectangular matrix (N < M)")
    MAT = np.random.rand(3, 5, 2)
    A = MAT[:, :, 0] + 1j*MAT[:, :, 1]
    Q, R = householder(A)
    perform_tests(A, Q, R, display=display, tolerance=tolerance)

    print("--------------------------")
    print("CASE 5: Taking a random LARGE COMPLEX rectangular matrix (N < M) with small entries")
    MAT = np.random.rand(100, 5, 2)/10000
    A = MAT[:, :, 0] + 1j*MAT[:, :, 1]
    Q, R = householder(A)
    perform_tests(A, Q, R, display=display, tolerance=tolerance)

    print("--------------------------")
    print("CASE 6: Taking a random REAL rectangular matrix with large entries")
    A = np.random.rand(100, 5)*10000
    Q, R = householder(A)
    perform_tests(A, Q, R, display=display, tolerance=tolerance)

    print("--------------------------")
    print("CASE 7: Taking a random REAL rectangular matrix with similar columns")
    # Creating a matrix with identical columns
    A = (np.ones((10, 20), dtype=np.float64) * np.random.rand(20)).T * 100
    # Introducing a small noise term to each column
    A += (np.random.rand(20, 10)/100000)
    Q, R = householder(A)
    perform_tests(A, Q, R, display=display, tolerance=tolerance)
    return


if __name__ == "__main__":
    print("Program Started.")
    # Insert keyword arguments below to customise display settings,
    # tolerance levels and numpy random seed.
    main()
    print("Program Finished.")
