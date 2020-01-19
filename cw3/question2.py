""" M4N9 Computational Linear Algebra - Project 3
Tudor Trita Trita
CID: 01199397

Implemented using Python 3.7 (need f-string python 3.6+)

Question 2. GMRES & Pre-Conditioning
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.linalg
import scipy.optimize
import scipy.sparse
import scipy.sparse.linalg


def construct_matrix_A(n):
    """Returns the matrix A (n^2, n^2) corresponding to Question 1."""
    n2 = n**2
    D0 = 4*np.ones(n2)  # 0th diagonal
    D1 = - np.ones(n2 - 1)  # -1st, 1st diagonals
    D1[n-1::n] = 0  # Setting every k*n-1 entries = 0 for k < n
    DN = - np.ones(n2 - n)  # -nth, nth diagonals
    return scipy.sparse.diags((DN, D1, D0, D1, DN), (-n, -1, 0, 1, n),
                              shape=(n2, n2), format="csr")


def construct_M_N(n):
    """ Returns matrices M & N in csr_matix format as described in Part 3."""
    n2 = n**2
    D0 = 2*np.ones(n2)  # 0th diagonal
    D1 = - np.ones(n2 - 1)  # -1st, 1st diagonals
    D1[n-1::n] = 0  # Setting every k*n-1 entries = 0 for k < n
    DN = - np.ones(n2 - n)  # -nth, nth diagonals
    return (scipy.sparse.diags((D1, D0, D1), (-1, 0, 1), shape=(n2, n2), format="csr"),
            scipy.sparse.diags((DN, D0, DN), (-n, 0, n), shape=(n2, n2), format="csr"))


def convergence_gmres_A():
    """ Function that demonstrates properties of convergence of GMRES
        method with the matrix A as specified in the question.

        Relevant function for part 2.
    """
    global conv_residuals
    def compute_residuals(r):
        """Helper function to retrieve residual + steps to convergence for
           GMRES operation in Scipy. Used as a callback function for
           scipy.sparse.linalg.gmres
        """
        global conv_residuals
        conv_residuals.append(r)
        return

    n_search = np.array([20, 40, 60, 80, 100, 120, 140, 160, 180])
    steps_till_conv_n = np.zeros(n_search.size)

    for i, n in enumerate(n_search):
        A = construct_matrix_A(n)
        # To average, we loop over 10 times
        for j in range(10):
            b = np.random.randn(n**2)
            conv_residuals = []
            x = scipy.sparse.linalg.gmres(A, b, callback=compute_residuals)
            steps_till_conv_n[i] += len(conv_residuals)

    # Divide by 10 to take the average:
    steps_till_conv_n /= 10

    fig220 = plt.figure(figsize=(13, 8))
    plt.plot(n_search, steps_till_conv_n)
    plt.xlabel("N")
    plt.ylabel("Steps Taken to Converge")
    plt.title("Figure 220 - Steps Taken for GMRES to Converge for Varying N",
              fontsize=13)
    plt.grid()
    plt.savefig("figures/figure220.png")
    plt.show()

    n_search = np.array([10, 50, 100, 150])

    fig221 = plt.figure(figsize=(13, 8))
    for i, n in enumerate(n_search):
        A = construct_matrix_A(n)
        b = np.random.randn(n**2)
        conv_residuals = []
        x = scipy.sparse.linalg.gmres(A, b, callback=compute_residuals)
        plt.semilogy(range(len(conv_residuals)), conv_residuals, label=f"N = {n}")

    plt.xlabel("Step Taken to Convergence")
    plt.ylabel("Residuals")
    plt.title("Figure 221 - GMRES Residuals for Varying N", fontsize=13)
    plt.legend()
    plt.grid()
    plt.savefig("figures/figure221.png")
    plt.show()
    return


def alternative_iterative_method(x0, n, gamma, b):
    """ Method described Part 3. Looping over k and i to explicitly show
        NxN independent tridiagonal systems using spsolve"""
    # Parameters:
    MAX_ITER = 1000
    n2 = n**2

    # Creating NxN versions of vector for easier indexing during iteration
    b = b.copy().reshape(n, n)
    b_transposed = b.copy().T
    x0 = x0.copy().reshape(n, n)
    x0_transposed = x0.copy().T
    x1 = x0.copy()
    x1_transposed = x0_transposed.copy()

    # No need for M, N, only a smaller tridiagonal system:
    H = scipy.sparse.diags((-1, 2, -1), (-1, 0, 1), shape=(n, n), format="csr")
    gammaI = scipy.sparse.diags((gamma,), (0,), shape=(n, n), format="csr")
    M1 = gammaI + H  # Corresponds to both (gI + M) & (gI + N) in equations
    M2 = gammaI - H  # Corresponds to both (gI - M) & (gI - N) in equations

    # Preallocating RHS of equations
    RHS7 = np.zeros((n, n), dtype=np.float64)
    RHS8 = np.zeros((n, n), dtype=np.float64)

    k = 0
    while k < MAX_ITER:
        for i in range(n):  # Loading RHS values for Equation (7):
            RHS7[:, i] = scipy.sparse.csr_matrix.dot(M2, x0_transposed[i]) + b_transposed[i]
        for i in range(n):  # Solving N independent tridig mat systems related to Eq(7):
            x1[i] = scipy.sparse.linalg.spsolve(M1, RHS7[i])
            RHS8[i] = scipy.sparse.csr_matrix.dot(M2, x1[i]) + b[i]  # Loading RHS values for Equation (8):
        for i in range(n):   # Solving N independent tridig mat systems related to Eq(8):
            x1_transposed[i] = scipy.sparse.linalg.spsolve(M1, RHS8[:, i])

        k += 1
        if np.allclose(x1_transposed, x0_transposed, rtol=1e-8):
            break
        x0_transposed = x1_transposed.copy()

    res = x1_transposed.T.reshape(n2)
    return res, k


def question26():
    """ Code used in calculations for Question 2.6."""
    n = 10
    n2 = n**2
    A = construct_matrix_A(n)
    x0 = np.random.randn(n2)
    b = np.random.randn(n2)

    # Compute optimal gamma:
    M, N = construct_M_N(n)

    # Eigenvalues of M and N are the same, so just use M for this now
    mu_max = scipy.sparse.linalg.eigsh(M, k=1, which='LM', return_eigenvectors=False)[0]
    mu_min = scipy.sparse.linalg.eigsh(M, k=1, which='SM', return_eigenvectors=False)[0]

    optimal_gamma_theoretical = np.sqrt(mu_min * mu_max)

    # We now verify this using our code:
    gamma_search = np.linspace(0.1, 4, 500)
    iters_array = np.zeros(500, dtype=int)

    for i, g in enumerate(gamma_search):
        iters_array[i] = alternative_iterative_method(x0, n, g, b)[1]

    min_graph = np.argmin(iters_array)
    min_iter = np.min(iters_array)
    min_gamma = gamma_search[min_graph]

    fig260 = plt.figure(figsize=(13, 8))
    plt.plot(gamma_search, iters_array)
    plt.plot(min_gamma,  min_iter, 'ro',
             label=f"Theoretical Gamma = {optimal_gamma_theoretical:.3f}\n" \
                   f"Min Iterations at (Gamma={min_gamma:.3f}, Iters={min_iter})")
    plt.axvline(x=optimal_gamma_theoretical)
    plt.legend()
    plt.grid()
    plt.xlabel("Gamma")
    plt.ylabel("Iterations til Convergence")
    plt.title("Figure 260 - Convergence Steps for Varying Gamma (N=10)")
    plt.savefig("figures/figure260.png")
    plt.show()
    return


def question27():
    """ Function containing code used for Part 7 of Question2."""
    global conv_residuals
    def catch(r):
        """Helper function to retrieve residual + steps to convergence for
           GMRES operation in Scipy. Used as a callback function for
           scipy.sparse.linalg.gmres
        """
        global conv_residuals
        conv_residuals.append(r)
        return

    def iterate(rk):
        """ Preconditioner Function for GMRES."""
        y = scipy.sparse.linalg.spsolve(P1, rk)
        RHS = scipy.sparse.csr_matrix.dot(P4, y) + rk
        zk = scipy.sparse.linalg.spsolve(P3, RHS)
        return zk


    N_search = np.array([20, 40, 60, 80, 100, 120, 140, 160, 180])
    steps_till_conv_N = np.zeros(N_search.size)

    fig271 = plt.figure(figsize=(13, 8))

    for i, n in enumerate(N_search):
        n2 = n**2
        A = construct_matrix_A(n)
        b = np.random.randn(n2)
        M, N = construct_M_N(n)
        mu_max = scipy.sparse.linalg.eigs(M, k=1, which='LM', return_eigenvectors=False)[0].real
        mu_min = scipy.sparse.linalg.eigs(M, k=1, which='SM', return_eigenvectors=False)[0].real
        gamma = np.sqrt(mu_max*mu_min)
        gammaI = scipy.sparse.diags((gamma,), (0,), shape=(n2, n2), format="csr")
        P1 = gammaI + M
        P2 = gammaI - N
        P3 = gammaI + N
        P4 = gammaI - M
        M = scipy.sparse.linalg.LinearOperator((n2, n2), matvec=iterate)
        conv_residuals = []
        x = scipy.sparse.linalg.gmres(A, b, M=M, callback=catch)
        steps_till_conv_N[i] += len(conv_residuals)
        n_steps = len(conv_residuals)
        plt.semilogy(range(n_steps), conv_residuals, label=f"N = {n}")

    plt.xlabel("Steps Required for Convergence")
    plt.ylabel("Residuals")
    plt.title("Figure 271 - GMRES + Preconditioner Residuals for Varying N", fontsize=13)
    plt.legend()
    plt.grid()
    plt.savefig(f"figures/figure271.png")
    plt.show()


    fig270 = plt.figure(figsize=(13, 8))
    plt.plot(N_search, steps_till_conv_N)
    plt.xlabel("N")
    plt.ylabel("Steps until convergence")
    plt.title("Figure 270 - GMRES + Preconditioner Convergence Required for Varying N", fontsize=13)
    plt.grid()
    plt.savefig(f"figures/figure270.png")
    plt.show()
    return


def main():
    # Perform Study of Convergence as in Question 2:
    print("\nPlots and output for part 2.2\n")
    convergence_gmres_A()
    print("\nPlots and output for part 2.6\n")
    question26()
    print("\nPlots and output for part 2.7\n")
    question27()
    return


if __name__ == "__main__":
    print("Starting CW3 Question 2\n")
    main()
    print("\nProgram Terminated")
