""" M4N9 Computational Linear Algebra - Mastery Project
Tudor Trita Trita
CID: 01199397

Implemented using Python 3.7 (need f-string python 3.6+)
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.linalg
import scipy.optimize
import scipy.sparse
import scipy.sparse.linalg


def construct_matrix_A(n, format="csr"):
    """Returns the matrix A (N^2, N^2) corresponding to Question 2."""
    inv_deltax_squared = (n+1)**2
    n2 = n**2
    D0 = 4*np.ones(n2)  # Main diagonal
    D1 = - np.ones(n2 - 1)  # -1st, 1st diagonals
    D1[n-1:n] = 0
    D2 = - np.ones(n2 - n)  # -nth, nth diagonals
    return inv_deltax_squared * scipy.sparse.diags((D2, D1, D0, D1, D2),
            (-n, -1, 0, 1, n), shape=(n2, n2), format=format, dtype=np.float64)


def arnoldi_iteration(A, v):
    """ Performs Arnoldi Iteration of a matrix given an initial condition.
        Returns:
            Vm: Matrix containing a set of orthonormal vector as its columns.
            H: Upper-Hessenberg matrix
    """
    m = v.size
    Vm = np.zeros((m, m+1), dtype=np.float64)  # Allocate memory for basis
    Vm[:, 0] = v / np.linalg.norm(v)
    H = np.zeros((m + 1, m), dtype=np.float64)
    for j in range(m):
        w = scipy.sparse.csr_matrix.dot(A, Vm[:, j])  # STEP 1 of algorithm
        for i in range(j+1):  # j+1 due to Python indexing
            H[i, j] = np.dot(w, Vm[:, i])  # STEP 2.1 of algorithm
            w -= H[i, j] * Vm[:, i]  # STEP 2.2 of algorithm
        H[j + 1, j] = np.linalg.norm(w)  # STEP 3.1 of algorithm
        Vm[:, j + 1] = w / H[j + 1, j]  # STEP 3.2 of algorithm
    V = Vm[:, :-1]
    H = H[:-1, :]
    return V, H


def heat_equation_arnoldi(v0, T):
    """ Computes solution to the heat equation using method described in the
        paper Gallapoulos and Saad (1992).
    """
    m = v0.size
    n = np.sqrt(m).astype(int)
    A = construct_matrix_A(n)
    V, H = arnoldi_iteration(A, v0)
    beta = np.linalg.norm(v0)
    return (beta * (V @ scipy.linalg.expm(-T*H)))[:, 0]


def heat_equation_expm(v0, T):
    """ Computes solution to the heat equation using direct sparse exponentiation
        using scipy's own sparse matrix exponentiation method.
    """
    m = v0.size
    n = np.sqrt(m).astype(int)
    A = construct_matrix_A(n, format="csc")  # "csc" as it is prefered for sparse expm
    return scipy.sparse.linalg.expm(-T*A) @ v0


def heat_equation_RK2(v0, T, dt):
    """ Computes solution to the heat equation using explicit 2nd-order Runge-Kutta
        time integration.
    """
    m = v0.size
    n = np.sqrt(m).astype(int)
    A = construct_matrix_A(n)
    t_space = np.linspace(0, T, 1 + T/dt)
    half_dt_A = - (dt/2) * A
    dt_A = - dt * A
    U = v0.copy()
    for i, t in enumerate(t_space[1:]):
        U_hlf = U + scipy.sparse.csr_matrix.dot(half_dt_A, U)
        U += scipy.sparse.csr_matrix.dot(dt_A, U_hlf)
    return U


def initial_conditions(n, params):
    """ Generates Gaussian initial Conditions."""
    # Co-ordinate grid
    x0 = np.tile(np.linspace(0, 1, n), n)
    x0 = np.reshape(x0, (n, n))
    y0 = np.tile(np.linspace(0, 1, n), n)
    y0 = np.reshape(y0, (n, n)).T[::-1]
    s, mu = params
    v0 = np.exp(-(x0-0.5)**2 / 2*s**2  - (y0-0.5)**2 / 2*s**2)
    return np.reshape(v0, n**2)


def main():
    # Plot 1:
    print("Plot 1: Differences in Norms")
    m = 10
    norms1, norms2, norms3 = [], [], []
    T_range = np.array([0.001, 0.01, 0.1, 0.5, 1, 2])
    v0_gaussian = initial_conditions(m, (5, 0.5))
    for i, T in enumerate(T_range):
        VT_arnoldi = heat_equation_arnoldi(v0_gaussian, T)
        VT_RK2 = heat_equation_RK2(v0_gaussian, T, 1e-6)
        VT_EXPM = heat_equation_expm(v0_gaussian, T)
        norms1.append(np.linalg.norm(VT_arnoldi - VT_RK2))
        norms2.append(np.linalg.norm(VT_arnoldi - VT_EXPM))

    fig1 = plt.figure(figsize=(13, 8))
    plt.loglog(T_range, norms1, 'r--', label=r"$|U_{AI} - U_{RK2}|$")
    plt.loglog(T_range, norms2, 'b', label=r"$|U_{AI} - U_{EXPM}|$")
    plt.xlabel('T')
    plt.ylabel("Difference in Norms")
    plt.title("Figure 1 - Plot of in Convergence of Different Methods (N=10)")
    plt.legend()
    plt.grid()
    plt.savefig("figures/fig1.png")
    plt.show()
    return


if __name__ == "__main__":
    print("Starting Output for Mastery Project\n")
    main()
    print("\nProgram Terminated")
