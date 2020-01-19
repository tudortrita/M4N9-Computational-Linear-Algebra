""" M4N9 Computational Linear Algebra - Project 3
Tudor Trita Trita
CID: 01199397

Implemented using Python 3.7 (need f-string python 3.6+)

Question 3. Image Denoising Algorithms
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.linalg
import scipy.optimize
import scipy.sparse
import scipy.sparse.linalg


def initialise_images(N, number=1, noisy=True):
    """ Function that Initialises Images for Question 3."""
    IMAGE = np.zeros((N+2, N+2))
    if number == 1:
        # Standard Image as Specified in Question 3
        B1 = int(np.floor(N*0.75 - 1e-12))  # Add some noise to not be singular
        B2 = B1 + 2
        B1 = N - B1
        IMAGE[B1:B2, B1:B2] = 1

    elif number == 2:
        # Modified Image: CROSS
        B11 = int(np.floor(N*0.6 - 1e-12))
        B12 = B11 + 2
        B11 = N - B11
        B21 = int(np.floor(N*0.9 - 1e-12))
        B22 = B21 + 2
        B21 = N - B21

        IMAGE[B11:B12, B21:B22] = 1
        IMAGE[B21:B22, B11:B12] = 1

    elif number == 3:
        # Modified Image: Combination of 1 and 2
        B1 = int(np.floor(N*0.75 - 1e-12))
        B2 = B1 + 2
        IMAGE[N-B1:B2, N-B1:B2] = 1

        B11 = int(np.floor(N*0.6 - 1e-12))
        B12 = B11 + 2
        B11 = N - B11

        B21 = int(np.floor(N*0.9 - 1e-12))
        B22 = B21 + 2
        B21 = N - B21

        IMAGE[B11:B12, B21:B22] = 1
        IMAGE[B21:B22, B11:B12] = 1

    elif number == 4:
        # Modified Image: TRIANGLE
        BM = int(np.ceil(N*0.5))
        B1 = int(np.floor(N*0.7))
        B2 = B1 + 2
        B1 = N - B1
        BR1 = np.arange(B1, B2)
        BR2 = np.arange(B1 - (BM - B1), B2 + (B2 - BM))

        for idx, Hpixel in enumerate(BR2):
            for idy, Vpixel in enumerate(BR1):
                if abs(BM - Hpixel) < idy + 1:
                    IMAGE[Vpixel, Hpixel] = 1

    else:
        return None
    if noisy:
        IMAGE[1:-1, 1:-1] += np.random.normal(loc=0.0, scale=0.1, size=(N, N))
        IMAGE[IMAGE > 1] -= 2*(IMAGE[IMAGE > 1] - 1)
        IMAGE[IMAGE < 0] *= -1
    return IMAGE


def construct_modified_A_N_M(n, mu):
    n2 = n**2
    deltax = 1 / (n + 1)
    deltax2 = deltax ** 2
    mu_deltax2 = mu*deltax2
    D01 = 4*np.ones(n2) + mu_deltax2
    D02 = D01 / 2
    D1 = - np.ones(n2-1)
    D1[n-1::n] = 0
    DN = -np.ones(n2-n)
    A = scipy.sparse.diags((DN, D1, D01, D1, DN), (-n, -1, 0, 1, n), format="csr")
    M = scipy.sparse.diags((D1, D02, D1), (-1, 0, 1), format="csr")
    N = scipy.sparse.diags((DN, D02, DN), (-n, 0, n), format="csr")
    return A, M, N


def denoise_GMRES(IM, n, mu):
    n2 = n**2
    deltax = 1 / (n + 1)
    deltax2 = deltax ** 2
    mu_deltax2 = mu*deltax2

    A, M, N = construct_modified_A_N_M(n, mu)

    I_IN = IM[1:-1, 1:-1]
    b = mu_deltax2 * np.reshape(I_IN.T, n2)

    mu_max = scipy.sparse.linalg.eigs(M, k=1, which='LM', return_eigenvectors=False)[0].real
    mu_min = scipy.sparse.linalg.eigs(M, k=1, which='SM', return_eigenvectors=False)[0].real
    gamma = np.sqrt(mu_max*mu_min)
    gammaI = scipy.sparse.diags((gamma,), (0,), shape=(n2, n2), format="csr")
    P1 = gammaI + M
    P2 = gammaI - N
    P3 = gammaI + N
    P4 = gammaI - M

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

    M = scipy.sparse.linalg.LinearOperator((n2, n2), matvec=iterate)

    conv_residuals = []
    x, _ = scipy.sparse.linalg.gmres(A, b, M=M, callback=catch)
    n_steps = len(conv_residuals)

    image_denoised_inner = np.reshape(x, (n, n)).T
    image_denoised = np.zeros((n+2, n+2))
    image_denoised[1:-1, 1:-1] = image_denoised_inner
    return image_denoised, n_steps


def question32():
    # Plotting Relationship between mu and GMRES n_iterations:
    n = 20
    nt = 1000
    n_iters_gmres = np.zeros(nt, dtype=int)
    mu_space = np.logspace(-2, 4, nt)
    for i, mu in enumerate(mu_space):
        IM = initialise_images(n, number=1)
        _, kD = denoise_GMRES(IM, n, mu)
        n_iters_gmres[i] = kD

    fig330 = plt.figure(figsize=(10, 7))
    plt.semilogx(mu_space, n_iters_gmres)
    plt.xlabel(r"$\mu$")
    plt.ylabel(r"Iterations till Convergence")
    plt.title(r"Figure 320 - Iterations Required for GMRES convergence vs $\mu$")
    plt.savefig("figures/figure320.png")
    plt.show()

    # Visualising Denoising Algorithm:
    n = 50
    IM = initialise_images(n, number=1)
    I1, _ = denoise_GMRES(IM, n, 1)
    I2, _ = denoise_GMRES(IM, n, 10)
    I3, _ = denoise_GMRES(IM, n, 100)
    I4, _ = denoise_GMRES(IM, n, 200)
    I5, _ = denoise_GMRES(IM, n, 500)
    I6, _ = denoise_GMRES(IM, n, 5000)

    fig321, ax321 = plt.subplots(nrows=3, ncols=2, figsize=(15, 15))

    ax321[0, 0].pcolor(I1, cmap='gray', vmin=0, vmax=1)
    ax321[0, 1].pcolor(I2, cmap='gray', vmin=0, vmax=1)
    ax321[1, 0].pcolor(I3, cmap='gray', vmin=0, vmax=1)
    ax321[1, 1].pcolor(I4, cmap='gray', vmin=0, vmax=1)
    ax321[2, 0].pcolor(I5, cmap='gray', vmin=0, vmax=1)
    ax321[2, 1].pcolor(I6, cmap='gray', vmin=0, vmax=1)


    ax321[0, 0].set_title(r"Denoised GMRES - $\mu = 1$")
    ax321[0, 1].set_title(r"Denoised GMRES - $\mu = 10$")
    ax321[1, 0].set_title(r"Denoised GMRES - $\mu = 100$")
    ax321[1, 1].set_title(r"Denoised GMRES - $\mu = 200$")
    ax321[2, 0].set_title(r"Denoised GMRES - $\mu = 500$")
    ax321[2, 1].set_title(r"Denoised GMRES - $\mu = 5000$")

    plt.suptitle(rf"Figure 321 - Visualising GMRES Denoising Method (N=50)", fontsize=14, y=0.92)
    plt.savefig("figures/figure321.png")
    plt.show()
    return


def shrink(X, gamma):
    absX = np.abs(X)
    X_shrinked = (X * np.maximum(absX - gamma, np.zeros(absX.shape))) / absX
    return X_shrinked


def denoise_iterative_method(IM, n, lambd, mu):
    MAX_ITER = 1000
    TOL = 10**-5
    n2 = n**2

    # Image
    phat = np.reshape(IM[1:-1, 1:-1].T, n2)

    # Initialising Variables
    P1 = np.zeros(n2)
    DH = np.zeros(n2)  # May need to be nxn+1
    DV = np.zeros(n2)  # May need to be nxn+1
    BH = np.zeros(n2)  # May need to be nxn+1
    BV = np.zeros(n2)  # May need to be nxn+1

    # Precomputing Variables:
    inv_lambda = 1/lambd
    deltax = 1 / (n + 1)
    deltax_2 = deltax ** 2
    inv_deltax = 1 / deltax
    mu_deltax_2 = mu*deltax_2
    lambda_deltax = lambd * deltax

    # Initialising
    D0 = (4*lambd + mu_deltax_2) * np.ones(n2)
    D1 = - lambd * np.ones(n2-1)
    D1[n-1::n] = 0
    DN = - lambd * np.ones(n2-n)

    A = scipy.sparse.diags((DN, D1, D0, D1, DN), (-n, -1, 0, 1, n), format="csr")

    # Helper variables for Parts B, C and D
    D0 = np.ones(n2)
    D1 = np.ones(n2-1)
    D1[n-1::n] = 0
    DN = np.ones(n2-n)
    HBC1 = scipy.sparse.diags((-D0, D1), (0, 1), format="csr")
    HBC2 = scipy.sparse.diags((-D0, DN), (0, n), format="csr")
    HD1 = scipy.sparse.diags((D0, -D1), (0, -1), format="csr")
    HD2 = scipy.sparse.diags((D0, -DN), (0, -n), format="csr")


    # Initial RHS (only mu * deltax2 * phat)
    b0 = mu_deltax_2 * phat
    b = b0.copy()

    k = 1
    while k < MAX_ITER:
        P2 = scipy.sparse.linalg.spsolve(A, b)

        if np.allclose(P2, P1):
            break

        P1 = P2.copy()
        k += 1

        # STEP B: Calculate values of DH and DV
        HBC1PDX = HBC1@P2 * inv_deltax
        HBC2PDX = HBC2@P2 * inv_deltax

        DH = shrink(BH + HBC1PDX, inv_lambda)
        DV = shrink(BV + HBC2PDX, inv_lambda)

        # STEP C: Calculate values of BH and BV
        BH += HBC1PDX - DH
        BV += HBC2PDX - DV

        # STEP D: Recompute RHS FOR STEP A:
        b = b0 - lambda_deltax * (HD1 @ (DH - BH) + HD2 @ (DV - BV))

    IM_Denoised = np.zeros((n+2, n+2))
    IM_Denoised[1:-1, 1:-1] = np.reshape(P2, (n, n)).T
    return IM_Denoised, k


def question33():
    # Visualising Denoising Algorithm:
    n = 40
    IM = initialise_images(n, number=1)
    I1, _ = denoise_iterative_method(IM, n, 0.1, 10)
    I2, _ = denoise_iterative_method(IM, n, 0.3, 20)
    I3, _ = denoise_iterative_method(IM, n, 2, 200)
    I4, _ = denoise_iterative_method(IM, n, 20, 100)
    I5, _ = denoise_iterative_method(IM, n, 15, 500)
    I6, _ = denoise_iterative_method(IM, n, 10, 500)

    fig331, ax331 = plt.subplots(nrows=3, ncols=2, figsize=(15, 15))

    ax331[0, 0].pcolor(I1, cmap='gray', vmin=0, vmax=1)
    ax331[0, 1].pcolor(I2, cmap='gray', vmin=0, vmax=1)
    ax331[1, 0].pcolor(I3, cmap='gray', vmin=0, vmax=1)
    ax331[1, 1].pcolor(I4, cmap='gray', vmin=0, vmax=1)
    ax331[2, 0].pcolor(I5, cmap='gray', vmin=0, vmax=1)
    ax331[2, 1].pcolor(I6, cmap='gray', vmin=0, vmax=1)


    ax331[0, 0].set_title(r"Denoised Iterative Algorithm - $\lambda = 0.1, $\mu = 10$")
    ax331[0, 1].set_title(r"Denoised Iterative Algorithm - $\lambda = 0.3, $\mu = 20$")
    ax331[1, 0].set_title(r"Denoised Iterative Algorithm - $\lambda = 2, $\mu = 200$")
    ax331[1, 1].set_title(r"Denoised Iterative Algorithm - $\lambda = 20, $\mu = 100$")
    ax331[2, 0].set_title(r"Denoised Iterative Algorithm - $\lambda = 15, $\mu = 500$")
    ax331[2, 1].set_title(r"Denoised Iterative Algorithm - $\lambda = 10, $\mu = 500$")

    plt.suptitle(rf"Figure 331 - Visualising Iterative Algorithm Denoising Method (N=50)", fontsize=14, y=0.92)
    plt.savefig("figures/figure331.png")
    plt.show()


    # Plotting Relationship between mu and GMRES n_iterations:
    n = 20
    nt = 100
    n_iters_gmres = np.zeros(nt, dtype=int)
    mu_space = np.logspace(-1, 4, nt)
    for i, mu in enumerate(mu_space):
        IM = initialise_images(n, number=1)
        _, kD = denoise_iterative_method(IM, n, 10, mu)
        n_iters_gmres[i] = kD

    fig332 = plt.figure(figsize=(10, 7))
    plt.semilogx(mu_space, n_iters_gmres)
    plt.xlabel(r"$\mu$")
    plt.ylabel(r"Iterations till Convergence")
    plt.title(r"Figure 332 - Iterations Required for Alternative convergence vs $\mu$ ($\lambda$=10$)")
    plt.savefig("figures/figure332.png")
    plt.show()

    n = 20
    nt = 100
    n_iters_gmres = np.zeros(nt, dtype=int)
    l_space = np.logspace(-1, 2, nt)
    for i, l in enumerate(l_space):
        IM = initialise_images(n, number=1)
        _, kD = denoise_iterative_method(IM, n, l, 100)
        n_iters_gmres[i] = kD

    fig333 = plt.figure(figsize=(10, 7))
    plt.semilogx(mu_space, n_iters_gmres)
    plt.xlabel(r"$\mu$")
    plt.ylabel(r"Iterations till Convergence")
    plt.title(r"Figure 333 - Iterations Required for Alternative convergence vs $\lambda$ ($\mu=100$)")
    plt.savefig("figures/figure333.png")
    plt.show()
    return


def main():
    # PART 3.1
    # Visualising first image with and without noise:
    n = 3
    I1_no_noise = initialise_images(n, noisy=False)
    I1_noisy = initialise_images(n, noisy=True)

    fig310, ax310 = plt.subplots(nrows=1, ncols=2, figsize=(13, 5))
    ax310[0].pcolor(I1_no_noise, cmap='gray', vmin=0, vmax=1)
    ax310[1].pcolor(I1_noisy, cmap='gray', vmin=0, vmax=1)

    ax310[0].set_title("Image Without Noise")
    ax310[1].set_title("Image With Noise")

    plt.suptitle("Figure 310 - Images as Specified in Part 1", fontsize=14)
    plt.savefig("figures/figure310.png")
    plt.show()

    # Part 3.2:
    print("\n\nFigures and Output for Part 3.2: \n\n")
    question32()
    # Part 3.3:
    print("\n\nFigures and Output for Part 3.3: \n\n")
    question33()
    return



if __name__ == "__main__":
    print("Starting CW3 Question 3\n")
    main()
    print("\nProgram Terminated")
