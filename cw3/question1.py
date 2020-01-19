""" M4N9 Computational Linear Algebra - Project 3
Tudor Trita Trita
CID: 01199397

Implemented using Python 3.7 (need f-string python 3.6+)

Question 1: Eigenvalue Algorithms
"""
import matplotlib.pyplot as plt
import numpy as np

import scipy
import scipy.sparse
import scipy.sparse.linalg

import warnings
warnings.filterwarnings("error")


def rayleigh_quotient_iteration(A, x, TOL=1e-10, MAX_ITER=20, display=True):
    """ Computes the Rayleigh Quotient Iteration for matrix A, vector b."""
    A = scipy.sparse.csr_matrix(A, dtype=np.float64)
    N = A.shape[0]
    accuracy_array = []  # To store differences of RQ's at each iteration
    RQ_array = []
    v = x / np.linalg.norm(x)  # Normalising initial vector
    RQ0 = v.dot(scipy.sparse.csr_matrix.dot(A, v))  # Initial Rayleigh Quotient
    RQ1 = RQ0
    for k in range(MAX_ITER):
        LHS = A - scipy.sparse.diags((RQ0, ), (0, ), shape=(N, N))
        try:
            w = scipy.sparse.linalg.spsolve(LHS, v)
        except:
            if display:
                print("\tWARNING!!\n\tMatrix is Singular. Solution has Converged!")
            break
        inv_norm_w = 1/np.linalg.norm(w)
        v = w * inv_norm_w
        RQ1 = v.dot(scipy.sparse.csr_matrix.dot(A, v))  # Computing new RQ

        # End-of-Iteration Checking Convergence, Appending values, etc.
        accuracy_array.append(np.abs(RQ1 - RQ0))
        RQ_array.append(RQ1)
        if abs(RQ1 - RQ0) < TOL:
            return RQ1, v, np.array(accuracy_array), np.array(RQ_array), k
        else:
            RQ0 = RQ1  # Updating 'old' Rayleigh Quotient
    if display:
        print(f"\tAlgorithm didn't Converge in {MAX_ITER} iterations!\n\n")
    return RQ1, v, np.array(accuracy_array), np.array(RQ_array), k


def question13():
    """ Code & Output used for Question 1.3."""
    np.set_printoptions(3)
    print("OUTPUT FOR PART 1.3:\n")
    N = 5  # Fix N=5 for this part
    A = scipy.sparse.diags((-1, 2, -1), (-1, 0, 1), shape=(N, N), format='csr')

    eigvalsA, eigvecsA = np.linalg.eig(A.todense())
    eigvecsA = np.asarray(eigvecsA)
    print(f"Eigenvalues of A: {eigvalsA}")
    print(f"Eigenvectors of A:\n{eigvecsA}\n")

    def display_cases(x, number):
        RQ, v, acc_array, rq_array, k = rayleigh_quotient_iteration(A, x)
        print(f"Case {number}: x = {x} \n")
        print(f"\tRQ = {RQ:.6f}")
        print(f"\tV = {v}")
        print(f"\tNo. Iterations till Convergence = {v}")
        print(f"\tArray of accuracy attained at each Iteration:\n\t{acc_array}")
        np.set_printoptions(10)
        print(f"\tRaleigh Quotients:\n\t{rq_array}")
        np.set_printoptions(3)
        return RQ, v, acc_array, rq_array, k

    x1 = np.array([1, 1, 1, 1, 1])
    display_cases(x1, 1)

    x2 = np.array([-1, 4, -2, 3, 10])
    acc_array2 = display_cases(x2, 2)[2]

    x3 = np.asarray(eigvecsA)[:, 3]
    display_cases(x3, "3: Fourth eigenvector of A\n\t")

    x4 = np.asarray(eigvecsA)[:, 3] + np.random.randn(5)/10
    display_cases(x4, "4: Fourth eigenvector of A + some noise\n\t")

    x5 = np.asarray(eigvecsA)[:, 3] + np.random.randn(5)
    acc_array5 = display_cases(x5, "5: Fourth eigenvector of A + some more noise\n\t")[2]
    print("FINISHED OUTPUT FOR PART 1.3\n")

    fig130 = plt.figure(figsize=(13, 8))
    plt.semilogy(range(1, 1+acc_array2.size), acc_array2, label="Residuals Case 2")
    plt.semilogy(range(1, 1+acc_array5.size), acc_array5, label="Residuals Case 5")
    plt.grid()
    plt.legend()
    plt.title("Figure 130 - Residuals for Cases 2 and 5 Showing Cubic Convergence ")
    plt.xlabel("Iteration k")
    plt.ylabel(r"Residual $|RQ_{t} - RQ_{t-1}|$")
    plt.savefig("figures/figure130.png")
    plt.show()
    return


def experiment(N, eigen_vec_index_select1, eigen_vec_index_select2, display=True):
    A = scipy.sparse.diags((-1, 2, -1), (-1, 0, 1), shape=(N, N), format='csr')
    eigvalsA, eigvecsA = np.linalg.eig(A.todense())
    eigvecsA = np.asarray(eigvecsA)

    v0 = eigvecsA[:, eigen_vec_index_select1]
    lambda0 = eigvalsA[eigen_vec_index_select1]

    # Take another eigenvector as the orthogonal direction
    z = eigvecsA[:, eigen_vec_index_select2]

    # Store in v1
    v1 = v0.copy()

    # Compute RQ for v0
    RQ, v, acc_array, rq_array, k = rayleigh_quotient_iteration(A, v0, display=False)

    tau = np.min((np.linalg.norm(v1 - v0), np.linalg.norm(v1 + v0)))

    MAX_ITER = 1000
    k = 1
    while k < MAX_ITER:
        v1 += z/100  # Add a bit of distance in orthogonal direction (STEP 2)
        RQ, v, acc_array, rq_array, k = rayleigh_quotient_iteration(A, v1, display=False)
        if np.allclose(RQ, lambda0):
            k += 1
            continue  # GO BACK TO STEP 2
        else:
            break

    # RQ not equal to initial evalue anymore:
    tau = np.linalg.norm(v1 - v0)
    norms_array = np.zeros(N)

    for i in range(N):
        eigv = eigvecsA[:, i]
        # Storing norms of all eigenvectors:
        norms_array[i] = np.linalg.norm(v1 - eigv)

    index_min = np.argmin(norms_array)
    norms_min = np.min(norms_array)

    if display:
        print("\nStarting Experiment:")
        print(f"Final tau = {tau}")
        print(f"v0 = {v0}")
        print(f"v1 = {v1}\n")
        print("Experiment Result:\n\t")
    if index_min == eigen_vec_index_select1:
        res = 1
        if display:
            print("RESULT 1")
    elif index_min == eigen_vec_index_select2:
        res = 2
        if display:
            print("RESULT 2")
    else:
        if display:
            print("RESULT 3")
        res =  3
    return res, tau


def question14():
    RESULTS_ARRAY = np.zeros((4, 3))
    TAU_ARRAY = np.zeros(4)


    for i, N in enumerate([3, 5, 7, 9]):
        x_vecs = np.arange(N)
        y_vecs = np.arange(N)

        for j, x in enumerate(x_vecs):
            for k, y in enumerate(y_vecs):
                if x == y:
                    continue
                res, tau = experiment(N, x, y, display=False)
                RESULTS_ARRAY[i, res-1] += 1
                TAU_ARRAY[i] += tau

        # Average across number of iterations for each N
        TAU_ARRAY[i] /= N*(N-1)

    figure140, ax = plt.subplots(figsize=(13, 8))
    im = ax.imshow(RESULTS_ARRAY)
    ax.set_xticks(np.arange(3))
    ax.set_yticks(np.arange(4))
    ax.set_xticklabels([f"RESULT {i}" for i in range(3)])
    ax.set_yticklabels([f"N = {N}" for N in [3, 5, 7, 9]])
    title = f"Experiment Results\n" + rf"$\tau$={TAU_ARRAY.round(2)}"
    ax.set_title(title)
    for i in range(4):
        for j in range(3):
            text = ax.text(j, i, RESULTS_ARRAY[i, j], color="w")
    plt.savefig("figures/figure140.png")
    plt.show()
    return


def question15():
    N = 5
    np.set_printoptions(3)
    A = scipy.sparse.diags((-1, 2, -1), (-1, 0, 1), shape=(N, N), format='csr')

    Q, R = np.linalg.qr(A.todense())
    print(f"The Matrix Q: \n\n {Q}\n")
    print(f"The Matrix R: \n\n {R}\n")

    N = 7
    np.set_printoptions(3)
    A = scipy.sparse.diags((-1, 2, -1), (-1, 0, 1), shape=(N, N), format='csr')

    Q, R = np.linalg.qr(A.todense())
    print(f"The Matrix Q: \n\n {Q}\n")
    print(f"The Matrix R: \n\n {R}\n")
    return


def pureQR(A, rtol=1e-10):
    """ Implements Pure QR Algorithm to Matrix A."""
    MAX_ITER = 1000

    # Container to store diagonal entries of A at every step
    AC = A.copy()  # Copy A to different memory for modification
    diags_A = [np.diag(AC)]
    k = 1
    while k < MAX_ITER:
        Q, R = np.linalg.qr(AC)  # Compute QR
        AC = R @ Q  # Swap Q, R to get A' = RQ

        # Computing diagonals
        dA = np.diag(AC)

        # Condition for diagonals converging:
        conv_a = np.allclose(dA, diags_A[-1], rtol=rtol)
        diags_A.append(dA)

        if conv_a:  # Break algorithm if entries away from diagonal are 0
            break
        k += 1
    return AC, diags_A, k, Q, R


def question16():
    N = 5
    A = scipy.sparse.diags((-1, 2, -1), (-1, 0, 1), shape=(N, N), format="dense")
    Aconvgd, diags_A, k, Q, R = pureQR(A)
    print("\nQUESTION 1.6\n")
    print(f"The matrix A after pure QR:\n\n{Aconvgd}\n")
    print(f"Iterations taken = {k}\n")
    return


def shiftsQR(A, mu, rtol=1e-10):
    MAX_ITER = 1e4
    AC = A.copy()
    N = A.shape[0]
    diags_A = [np.diag(AC)]
    k = 1
    MU = scipy.sparse.diags((mu,), (0,), shape=(N, N), format='dense', dtype=np.float64)
    while k < MAX_ITER:
        Q, R = np.linalg.qr(AC - MU)  # SHIFTED QR(A - muI)
        AC = (R @ Q) + MU  # SHIFTED A = RQ + muI

        # Computing diagonals
        dA = np.diag(AC)

        # Condition for diagonals converging:
        conv_a = np.allclose(dA, diags_A[-1], rtol=rtol)
        diags_A.append(dA)

        if conv_a:  # Break algorithm if entries away from diagonal are 0
            break
        k += 1
    return AC, diags_A, k, Q, R


def question17():
    N = 5
    mu = 0.5
    A = scipy.sparse.diags((-1, 2, -1), (-1, 0, 1), shape=(N, N), format="dense")
    Aconvgd, diags_A, k, Q, R = shiftsQR(A, mu)
    print("\nQUESTION 1.7\n")
    print(f"The matrix A after shifted QR (mu=0.5):\n\n{Aconvgd}\n")
    print(f"Iterations taken = {k}\n")

    mu_range = np.linspace(-1, 5, 1000)
    k_array = np.zeros(mu_range.size)
    for i, mu in enumerate(mu_range):
        _, _, k, _, _ = shiftsQR(A, mu)
        k_array[i] = k

    k_min = np.min(k_array)
    mu_min = mu_range[np.argmin(k_array)]
    fig170 = plt.figure(figsize=(13, 8))
    plt.semilogy(mu_range, k_array)
    plt.semilogy(mu_min, k_min, 'ro', label=f"Min Iters at ({mu_min:.2f}, {k_min})")

    # N biggest points:
    N_k_max_id = np.argsort(k_array)[::-1][:N]
    N_k_max = np.sort(k_array)[::-1][:N]
    N_mu_max = mu_range[N_k_max_id]
    for i, j in zip(N_k_max, N_mu_max):
        plt.semilogy(j, i, 'o', label=rf"$\mu$ = {j:.2f}")
    plt.legend()
    plt.grid()
    plt.title(r"Figure 170 - Plot of Iterations till Convergence against $\mu$ + 5 largest k")
    plt.xlabel(r"$\mu$")
    plt.ylabel("Iterations till Convergence")
    plt.savefig("figures/figure170.png")
    plt.show()
    return


def question18():
    # Part 1.8 Exercises
    N = 5
    A = np.random.randn(N, N)
    S = A @ A.T

    # Taking diagonal values of S matrix
    SC, diags_S, ks, QS, RS = pureQR(S)

    # Creating S3
    S3 = S - np.triu(S, 2) - np.tril(S, -2)
    S3C, diags_S3, ks3, QS3, RS3 = pureQR(S3)

    print(f"The matrix S:\n\n{S}\n")
    print(f"The matrix S after pure QR:\n\n{SC}\n")
    print(f"Iterations taken = {ks}\n")
    print(f"The matrix S3:\n\n{S3}\n")
    print(f"The matrix S3 after pure QR:\n\n{S3C}\n")
    print(f"Iterations taken = {ks3}\n")

    # Finding Averages:
    iters_s = 0
    iters_s3 = 0
    for i in range(100):
        A = np.random.randn(N, N)
        S = A @ A.T

        # Taking diagonal values of S matrix
        iters_s += pureQR(S)[2]

        # Creating S3
        S3 = S - np.triu(S, 2) - np.tril(S, -2)
        iters_s3 += pureQR(S3)[2]

    iters_s /= 100
    iters_s3 /= 100
    print("Average Iterations for pure QR Matrix S after 100 times:")
    print(round(iters_s))
    print("Average Iterations for pure QR Matrix S3 after 100 times:")
    print(round(iters_s3))
    return


def main():
    np.set_printoptions(3)
    question13()  # Code and Output for part 1.3
    # question14()  # Code and Output for part 1.4, uncomment to run
    question15()  # Code and Output for part 1.5
    question16()  # Code and Output for part 1.6
    question17()  # Code and Output for part 1.7
    question18()  # Code and Output for part 1.8
    return


if __name__ == "__main__":
    print("Program Started\n")
    main()
    print("\nProgram Finished")
