import math
import numpy as np
import os

print("\nThe Row-reduced Echelon Form Calculator")

allowed_numbers = []
for i in range(20):
    allowed_numbers.append(str(i + 1))

r = input("Number of rows: ")

while r not in allowed_numbers:  # must input until valid character entered
    print("\nError: please enter a number from 1 to 20.")
    r = input("\nNumber of rows: ")

c = input("Number of columns: ")

while c not in allowed_numbers:
    print("\nError: please enter a number from 1 to 20.")
    c = input("Number of columns: ")

matrix_entries = input("Enter the " + str(int(r) * int(c)) + " elements of your matrix row-wise,\n"
                       + "separating each element with at least one space:\n")
a_lstring = [x for x in matrix_entries.split()]

while len(a_lstring) != int(r) * int(c):
    print("\nError: please follow the instructions.\n")
    matrix_entries = input("Enter the " + str(int(r) * int(c)) + " elements of your matrix row-wise,\n"
                           + "separating each element with at least one space:\n")
    a_lstring = [x for x in matrix_entries.split()]

lst = []
j = 0
while j <= len(a_lstring) - 1:
    if a_lstring[j].lstrip('-').replace('.', '').isdigit() == True:  # so that negative numbers are floats are no problem
        a = round(float(a_lstring[j]), 3)
        lst.append(a)
        j = j + 1
    else:
        print("\nError: please follow the instructions.\n")
        matrix_entries = input("Enter the " + str(int(r) * int(c)) + " elements of your matrix row-wise,\n"
                               + "separating each element with at  least one space:\n")
        a_lstring = [y for y in matrix_entries.split()]

arr = np.array(lst)
arr_2d = np.reshape(arr, (int(r), int(c)))
print("\nYour matrix is:\n")  # printing the matrix
for z in arr_2d:
    for n in z:
        print(n, end=" ")
    print()

zeros_list = []
for k in range(int(r)):
    zeros_list.append(0)

arr_zeros = np.array(zeros_list).T
arr_zeros_row = arr_zeros.T
columns, rows = int(c), int(r)

count = 0  # counts number of zero columns
for k in range(columns):
    if np.array_equal(arr_2d[:, k], arr_zeros):
        count = count + 1

q = 0
while q <= columns - 1:  # looping across columns
    if np.array_equal(arr_2d[:, q], arr_zeros):  # ignore zero columns
        q = q + 1
    else:
        for m in range(rows):
            w = m
            while w + count <= rows - 1:  # looping across rows in column
                if np.array_equal(arr_2d[w][q], arr_zeros[0]):
                    w = w + 1  # go to next row in column if element is 0
                else:
                    w_qfloat = float(arr_2d[w][q])  # to avoid floating point error
                    arr_2d[w, :] *= 1 / w_qfloat  # make leading entry equal to one
                    arr_2d[[m, w]] = arr_2d[[w, m]]  # swap mth row with wth row
                    g = 0
                    while g <= rows - 1:  # row op being performed on gth row
                        if w == g:
                            g = g + 1
                        else:
                            g_qfloat = float(arr_2d[g][q])  # avoid float point
                            arr_2d[g, :] -= g_qfloat * arr_2d[w, :]  # take away wth row from gth row
                            g = g + 1
                    break
            q = q + 1


print("\nYour matrix in row-reduced echelon form is:\n")  # printing the matrix
for d in arr_2d:
    for e in d:
        print(round(e + 0, 3), end=" ")
    print()
