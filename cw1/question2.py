""" M4N9 Computational Linear Algebra - Project 1
Tudor Trita Trita
CID: 01199397

Implemented using Python 3.7

Question 2. Polynomial fitting using QR Factorisation

Functions in this file:

1. back_substitution : solves Ax = b system
2. polynomial_fit : fits polynomial to data
3. calc_error : computes l2-norm error of polynomial fit
4. main : performs all calculations and printing
"""
import numpy as np
import question1

def back_substitution(A, b, M):
    """Solves equation Ax=b for x assuming A is an upper-triangular matrix.
    """
    x = np.zeros(M)  # Pre-allocating memory
    x[-1] = b[-1]/A[-1, -1]  # Solving last entry

    # Iterating through matrix backwards:
    for j in range(M - 2, -1, -1):
        x[j] = (b[j] - A[j, j+1:] @ x[j+1:]) / A[j, j]  # Back-substitution algorithm
    return x


def polynomial_fit(x, b, degree=3):
    """Fits a polynomial with coefficients vector c to some data b, x.
        Array containing polynomial coefficients of the form:
            [c_0, c_1, c_2, ..., c_{n-1}]
        for the polynomial:
            p(x) = c_0 + c_1*x + c_2*(x^2) + ... + c_{n-1}*x^{n-1}
    """
    N = x.size
    M = degree + 1

    # Creating vandermonde matrix of polynomials of data x
    A = np.vander(x, N=M, increasing=True)

    # Computing Q, R factorization of A
    Q, R = question1.householder(A)

    # Computing reduced Q, R - Qhat, Rhat
    Qhat, Rhat = Q[:, :M], R[:M, :]

    # Computing RHS of upper-triangular system Rhat*x = Qhatstar*b
    Qhat_times_b = (Qhat.T).dot(b)

    # Computing coefficients for the system of equations
    polynomial_coefficients = back_substitution(Rhat, Qhat_times_b, M)
    return polynomial_coefficients


def calc_error(c, x, b, degree=3):
    """ Calculates error of the polynomial fit.
    """
    M = degree + 1
    A = np.vander(x, N=M, increasing=True)
    error = np.sum(np.abs(A @ c - b)**2)
    return error


def main():
    DATA = np.genfromtxt('readings.csv', delimiter=',')
    x = DATA[:, 0]
    b = DATA[:, 1]

    # Setting maximum degree of polynomial to iterate over
    Nt = 20
    range_iterations = np.arange(1, Nt + 1, 1)

    # Preallocating matrix for errors:
    error_array = np.zeros(Nt)

    for i in range_iterations:
        p_coeff = polynomial_fit(x, b, degree=i)

        error = calc_error(p_coeff, x, b, degree=i)
        error_array[i - 1] = error

        if i == 1:
            improvement = np.nan
        else:
            improvement = -  (error_array[i - 1] - error_array[i - 2])/error_array[i - 2] * 100
        print("- Poly. d = %s. Error = %s. Improvement = %s percent." %(i, round(error, 8), round(improvement, 3)))

    return

if __name__ == "__main__":
    print("Program Started")
    main()
    print("Program Finished")
