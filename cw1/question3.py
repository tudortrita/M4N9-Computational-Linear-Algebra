""" M4N9 Computational Linear Algebra - Project 1
Tudor Trita Trita
CID: 01199397

Implemented using Python 3.7

Question 3. Analysis of readings2.csv data.
"""
import numpy as np
np.set_printoptions(precision=2)  # To display matrix in terminal nicely

import question1

A = np.genfromtxt('readings2.csv', delimiter=',')  # Importing our data
Q, R = question1.householder(A)

N, M = A.shape

Rhat = R[:M, :]
print("Reduced matrix R:")
print()
print(Rhat)


Rhat = question1.remove_small_entries(Rhat, accuracy=10**-12)
print("Reduced matrix R with small entries removed:")
print()
print(Rhat)
