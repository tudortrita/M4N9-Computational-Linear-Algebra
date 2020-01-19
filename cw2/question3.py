""" M4N9 Computational Linear Algebra - Project 2
Tudor Trita Trita
CID: 01199397

Implemented using Python 3.7 (need f-string python 3.6+)

Question 3. Exponential Integrators
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy
import scipy.linalg
import scipy.sparse.linalg
import time

import question2
import rexi_coefficients


class Computations:
    """ Contains Runge Kutta 2nd Order, REXI and Direct Solution."""
    def __init__(self, N=100, H=10):
        self.N = N
        self.H = H

        self.initial_condition()
        self.matrices()

    def initial_condition(self):
        self.dx = self.H / (self.N + 1)
        self.inv_dx = 1/self.dx
        self.inv_dx2 = self.inv_dx**2

        self.x = np.linspace(0, self.H, self.N, dtype=np.complex128)
        self.V0 = np.exp(-(self.x-5)**2/0.2) - np.exp(-125)
        self.W0 = np.zeros(self.N, dtype=np.complex128)
        self.U0 = np.concatenate((self.V0, self.W0))

    def matrices(self):
        """ Initialises the matrices L and K."""
        # Creating L
        L = scipy.sparse.diags((self.inv_dx2, -2*self.inv_dx2, self.inv_dx2, 1),
                                (-(self.N+1), -self.N, -(self.N-1), self.N),
                                shape=(2*self.N, 2*self.N), dtype=np.complex128)
        self.L = scipy.sparse.csr_matrix(L)
        self.L[-(self.N+1), 0], self.L[-1, -self.N] = 0, 0

        # Computing largest eigenvalue of L explicitely:
        self.mu_max = self.inv_dx*np.sqrt(2*(1 + np.cos(np.pi/(self.N+1))))

        # Creating K
        self.K = scipy.sparse.diags((-self.inv_dx2, 2*self.inv_dx2, -self.inv_dx2),
                                    (-1, 0, 1),  # Diagonals
                                    shape=(self.N, self.N),  # Size of matrix
                                    dtype=np.complex128)

    def EXPM(self, T):
        U = scipy.linalg.expm(T*self.L) @ self.U0
        return U

    def RK2(self, T=2.5, dt=0.00001):
        """ Computes Runge Kutta 2nd Order solution of the initial condition
            at time T with steps dt.
        """
        t_space = np.linspace(0, T, 1 + T/dt)
        U = self.U0.copy()
        half_dt_L = (dt / 2) * self.L
        dt_L = dt * self.L

        for i, t in enumerate(t_space[1:]):
            U_hlf = U + half_dt_L @ U
            U += dt_L @ U_hlf
        return U

    def REXI(self, T=2.5, h=None, M=None):
        """REXI Exponential Integrator Solution"""
        if h:
            M = 1.1*T*self.mu_max/h
        elif M:
            h = 1.1*T*self.mu_max/M
        else:
            raise Exception("ERROR: At least ONE of h or M needs to be assigned!")

        alpha, beta = rexi_coefficients.RexiCoefficients(h, M)
        J = alpha.shape[0]

        # Initialising variables V, W:
        V = np.zeros((self.N, J), dtype=np.complex128)
        W = np.zeros((self.N, J), dtype=np.complex128)

        # Pre-computing terms outside of the loop:
        LHS_pre = T**2 * self.K
        RHS_pre = self.W0*T
        for j in range(J):
            LHS = LHS_pre + scipy.sparse.diags([alpha[j]**2],
                                                offsets=0,
                                                shape=[self.N, self.N],
                                                dtype=np.complex128)
            RHS = alpha[j]*self.V0 - RHS_pre
            L, U = question2.bandedLU(LHS, 1, 1)
            RHS2 = scipy.sparse.linalg.spsolve_triangular(L, RHS, lower=True)
            # Can't use spsolve_triangular again as U isn't completely upper-triangular due to roundoff
            V[:, j] = scipy.sparse.linalg.spsolve(U, RHS2)
            W[:, j] = (self.V0 - alpha[j]*V[:, j])/T

        # End loop:
        Uj_mat = np.concatenate((V, W), axis=0)
        U = np.sum(np.multiply(Uj_mat, beta[None, :]), axis=1)
        return U, (h, M)

    def plot_solutions(self, solutions_list, plot_w=False, savefig_filename=None, display=True):
        """ Plots solutions for different times and methods.
        solutions_dict needs to be in the following format:
        [[solution1, [method1, T1]], [solution2, [method2, T2]], ...]
        Constrained to parameters the class was initialised with (N, H)
        """
        plt.figure(figsize=(13, 8))
        for s in solutions_list:
            U = s[0]
            method, T = s[1]
            plt.plot(self.x, U[:self.N], label=rf"$U$ : {method}, $T = {T}$")
            if plot_w:
                plt.plot(self.x, U[self.N:], label=rf"$U_t$ : {method}, $T = {T}$")
        plt.xlabel(r"$x$")
        plt.ylabel(r"$U, U_t$")
        plt.title("Plot of Various Models at Different Times")
        plt.legend()
        plt.grid()
        if savefig_filename:
            plt.savefig(savefig_filename) if savefig_filename.endswith(".png") else plt.savefig(savefig_filename+".jpg")
        if display:
            plt.show()

def main():
    # FIGURE 0: Showing Wave at Time 2.5 for REXI & RK2 (CAST TO REALS)
    C1 = Computations(N=100, H=10)
    U_REXI, _ = C1.REXI(T=2.5, M=50)
    U_RK2 = C1.RK2()
    C1.plot_solutions([[U_REXI, ['REXI', 2.5]],
                       [U_RK2, ['RK2', 2.5]]],
                       plot_w=True)

    # FIGURE 1: Contour plot of solution propagating through time (CAST TO REALS)
    C2 = Computations(N=100, H=10)
    times_array = np.linspace(0, 5)
    for i, T in enumerate(times_array[1:]):
        print(i)
        U_REXI, _ = C2.REXI(T=T, M=30)
        if i==0:
            U_REXI_MATRIX = np.concatenate([C2.U0[:, None], U_REXI[:, None]], axis=1)
        else:
            U_REXI_MATRIX = np.concatenate([U_REXI_MATRIX, U_REXI[:, None]], axis=1)

    fig1 = plt.figure(figsize=(13, 8))
    plt.contourf(C2.x, times_array, U_REXI_MATRIX[:C2.N, :].T, cmap=matplotlib.cm.jet)
    plt.colorbar()
    plt.xlabel("X Interval [0, H]")
    plt.ylabel("Time T")
    title1 = "Figure 1 - Countour plot of REXI Solution at Various Times"
    plt.title(title1)
    plt.savefig(title1+".png")
    plt.show()

    # FIGURE 2: Computing convergence between RK2 and REXI FOR varying M, dt:
    C3 = Computations(N=10, H=10)
    U_RK2_1 = C3.RK2(T=2.5, dt=pow(10, -4))
    U_RK2_2 = C3.RK2(T=2.5, dt=pow(10, -5))
    U_RK2_3 = C3.RK2(T=2.5, dt=pow(10, -6))

    norms1, norms2, norms3 = [], [], []
    range_M = np.linspace(10, 300, 30, dtype=int)
    for M in range_M:
        U_REXI, _ = C3.REXI(M=M)
        norm1 = np.linalg.norm(U_RK2_1 - U_REXI)
        norm2 = np.linalg.norm(U_RK2_2 - U_REXI)
        norm3 = np.linalg.norm(U_RK2_3 - U_REXI)

        norms1.append(norm1)
        norms2.append(norm2)
        norms3.append(norm3)

    fig2 = plt.figure(figsize=(13, 8))
    plt.semilogy(range_M, norms1, 'r--', label=rf"$\Delta t$={pow(10, -4)}")
    plt.semilogy(range_M, norms2, 'b', label=rf"$\Delta t$={pow(10, -5)}")
    plt.semilogy(range_M, norms3, 'k', label=rf"$\Delta t$={pow(10, -6)}")

    plt.xlabel('M')
    plt.ylabel("Difference in Norms")
    title2 = "Figure 2 - Plot of in Convergence for U_RK2 & U_REXI, varying M and dt (T=2.5)"
    plt.title(title2, fontsize=15)
    plt.legend()
    plt.grid()
    plt.savefig(title2+".png")
    plt.show()

    # FIGURE 3: Dependence of M, H for different M, N
    C4 = Computations(N=5)
    C5 = Computations(N=10)
    C6 = Computations(N=15)

    range_M = np.linspace(10, 300, 30, dtype=int)
    h1_array, h2_array, h3_array = [], [], []
    for M in range_M:
        U_REXI1, [h1, M] = C4.REXI(M=M)
        U_REXI2, [h2, M] = C5.REXI(M=M)
        U_REXI3, [h3, M] = C6.REXI(M=M)
        h1_array.append(h1)
        h2_array.append(h2)
        h3_array.append(h3)

    fig3 = plt.figure(figsize=(13, 8))
    plt.plot(range_M, h1_array, 'r--', label="REXI N=5")
    plt.plot(range_M, h2_array, 'b', label="REXI N=10")
    plt.plot(range_M, h3_array, 'k', label="REXI N=15")

    plt.xlabel('M')
    plt.ylabel('h')
    title3 = "Figure 3 - Plotting dependencies of h and M for different N"
    plt.title(title3, fontsize=15)
    plt.legend()
    plt.grid()
    plt.savefig(title3+".png")
    plt.show()
    return

if __name__ == "__main__":
    print("Program Started")
    print()
    main()
    print()
    print("Program Finished")
