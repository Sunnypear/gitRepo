# for i in range(1,10):
#     for j in range(1,10):
#         print(j)
#         if j == 3:
#             break
#
import numpy
# a = [1,2,3,4,5,6,7,8,9]
# for i,item in enumerate(a):
#     print("i=%d,item=%d"%(i,item))

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.utils.linear_assignment_ import linear_assignment


ndim = 4
H_TPM = np.zeros((ndim, 2 * ndim))
for i in range(ndim):
    H_TPM[i,2*i+1] = 1
print(H_TPM)
c = np.zeros((19,19))
a = np.array([[84, 65, 3, 34], [65, 56, 23, 35], [63, 18, 35, 12]])
print(a[0,0])
indexs = linear_assignment(a)
row,col = linear_sum_assignment(a,True)
print(row[:,np.newaxis])
index = np.c_[row[:,np.newaxis], col[:,np.newaxis]]
print(indexs)
print(index)
# print("行坐标:", row, "列坐标:", col, "最小组合:", a[row, col])
# row, col = linear_sum_assignment(a, True)
# print("行坐标:", row, "列坐标:", col, "最大组合:", a[row, col])

