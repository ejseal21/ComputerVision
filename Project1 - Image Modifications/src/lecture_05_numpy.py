'''
lecture05_numpy.py
Demo of Numpy Matrix features
Oliver W. Layton
CS251: Data Analysis and Visualization
'''

import numpy as np
import time

'''
Creating Numpy Matrices
'''

# Make a numpy Matrix from a 2D python list
mat = np.matrix([[1,2,3],[4,5,6],[7,8,9]])
print('Matrix\n', mat)
print('Type of matrix is\n',)


# Make a numpy array from a 2D python list
arr = np.array([[1,2],[3,4]])
print('Array\n', arr)

# Specify the type of the matrix to be of type float
matFloat = np.matrix([[1,2,3],[4,5,6],[7,8,9]])
print('Matrix float\n', matFloat)
print('Type of float matrix is\n', )

# Specify the type of the matrix to be of type string
matString = np.matrix([[1,2,3],[4,5,6],[7,8,9]])
print('Matrix string\n', matString)

# Convert back from matrix/ndarray to Python list
matAsList = mat.tolist()
print('Back as a Python list:\n', matAsList)

# IMPORTANT: Check the dimensions of a matrix
print('Dimensions of mat: ', mat.shape)

# Access 1st dim (#rows), 2nd dim (#cols) (Use f-string)
print(f'Num rows/cols in mat are {mat.shape[0]} / {mat.shape[1]}')

# Check number of elements total
print('Num elements in mat:', mat.size)

# Brief detour: In python you can replace the workflow of 
# list-building by creating an empty list and looping to append...
myList = []
for i in range(5):
    myList.append(i)
# print('myList build the usual way', myList)

# ...with Python list comprehensions
myListComp = [i for i in range(5)]

print('myListComp', myListComp)

# you can build lists using any function of i. How about i^2?
myListSqr = [i**2 for i in range(5)]
print('myListSqr', myListSqr)

'''
Basic Accessing and modifying matrices
'''

# Use square brackets to access an element in a matrix
# When indexing matrixes we ALWAYS use both [row, col]
# indices...neither one should ever be omitted!
print('Element 0,0 of mat is:', mat[0,0])

# Modifying single values is similar
# print('Mat is now:\n', mat)

# Make a row vector out of myListSqr, then convert to column vector.
# Also, print the dimensions
rowVec = np.matrix(myListSqr)
colVec = rowVec.T
print('Row vec:\n', rowVec, 'with dimensions', rowVec.shape)
print('Col vec:\n', colVec, 'with dimensions', colVec.shape)


# Numpy tries to be very memory efficient so when you use
# vec.T to transpose, it creates a SHALLOW COPY of vec!
# Changes to the transpose will affect the original matrix!!!
colVec[1] = 99
print('colVec after change\n', colVec)
print('rowVec after change\n', rowVec)

# To get a genuine DEEP COPY, use the copy() method
# print('colVec after copy()\n', colVec)
# print('rowVec after copy()\n', rowVec)

'''
Slicing matrices
'''

# Like 2D python lists, we can slice to get rows/cols
# using colon notation.
# Again: ALWAYS double index your matrices!
row1 = mat[0, :]
col2 = mat[:, 1]
print('mat:\n', mat)
print('row1 of mat:\n', row1)
print('col2 of mat:\n', col2)

# Replacing a row/column with a constant is easy
mat[0, :] = 5

# Replacing a row/column with another is also easy
mat[:, 0] = 5
print(mat)
# Note: Remember to check your dimensions or you may get errors
# Try: replace row 0 with new matrix 0,1,2
# Try: replace col 0 with new matrix 0,1,2
# print('After replacing row and col 0 with a 0,1,2:\n', mat)

'''
Stacking vectors
'''

# Use the np.vstack() method to concatenate
# two row 1xN vectors into a 2xN metrix
# We will use list comprehension to build this up quickly
# (there is np.hstack() to concatenate column vectors)
rowV1 = np.matrix([i for i in range(10)])
rowV2 = np.matrix([2*i for i in range(10)])
stackedMat = np.vstack([rowV1, rowV2])
print('Stacked mat:\n', stackedMat)

'''
Numpy matrix axes
'''

# For many numpy operations, we need to specify whether we
# will perform the computations on rows or columns in a matrix
# The axis optional parameter allows us to specify across
# rows(axis=0) or cols(axis=1)
#
# Example: np.diff() for subtracting successive pairs of elements
# across rows or corresponding elements across cols
# Let's do this on stackedMat
diff0 = np.diff(stackedMat, axis=0)
diff1 = np.diff(stackedMat, axis=1)
print('stackedMat:\n', stackedMat)
print('diff along axis0 is:\n', diff0)
print('diff along axis1 is:\n', diff1)

'''
Vectorized operations
'''


def timeit(fun):
    '''Just a function to time the runtime of another function'''
    def timer():
        start = time.time()
        fun()
        end = time.time()
        print(f'Took {end - start:.3} secs to run.')
    return timer


@timeit
def addOneLoop():
    '''Use for loop to take reciprocal of row vector'''
    longRow = np.array([i for i in range(1, 1000000)])
    for i in range(len(longRow)):
        longRow[i] = longRow[i] + 1


@timeit
def addOneVectorized():
    '''Vectorized version of taking the reciprocal of row vector'''
    longRow = np.array([i for i in range(1, 1000000)])
    longRow = longRow + 1

# Dynamic typing in python makes for loops with lots of small
# operations slow
# print('addOneLoop:')
# addOneLoop()

# Vectorization allows Numpy to stop searching at runtime
# and use efficient pre-compiled functions to batch-process
# the computation over the matrix
# print('addOneVectorized:')
# addOneVectorized()