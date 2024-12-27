import numpy as np
# a = [[1,2,3],[4,5,6],[7,8,9]]
# b = [[1,2,3],[4,5,6],[7,8,9]]
# a, b = np.asarray(a), np.asarray(b)
# # a: NM; b: LM
# # a2:N ; b2:L
# a2, b2 = np.square(a).sum(axis=1), np.square(b).sum(axis=1)
# # np.dot(a,b.T): L=NL
# # a2[:,None]: N1   b2[None:]: 1L
# # (a[i]-b[i])**2
# # 求每个embedding的平方和
# # sum(N) + sum(L) -2 x [NxM]x[MxL] = [NxL]
# temp = np.dot(a,b.T)
# temp1 = a2[:,None]
# temp2 = b2[None,:]
# r2 = -2. * np.dot(a, b.T) + a2[:, None] + b2[None, :]
# print(r2)

# a = [[1,2,3],
#      [4,5,6],
#      [7,8,9]]
# a = np.array(a)
# b = a.min(axis=0)
# c = a.min(axis=1)
# d = a
# print()
# a = [1,2,3,4]
# b = [5,6,7,8]
# c = [a,b]
# print(c[1][1:])
# e = [9,10,11,12,13,14,15,16,17,18,19]
#
# for i in range(1,len(e)):
#     print(e[i])
#
# print(e[1:])
# f = np.r_[a,e]
# c = np.r_[a,b]
# d = np.c_[a,b]
# print(c)
ndim = 4
dt = 1
F_matrix = np.eye(2 * ndim, 2 * ndim)
"""
[[1,0,0,0,0,0,0,0],
 [0,1,0,0,0,0,0,0],
 [0,0,1,0,0,0,0,0],
 [0,0,0,1,0,0,0,0],
 [0,0,0,0,1,0,0,0],
 [0,0,0,0,0,1,0,0],
 [0,0,0,0,0,0,1,0],
 [0,0,0,0,0,0,0,1]
]
x0 :[x,dx,y,dy,a,da,h,dh].T
"""
# for i in range(ndim):
#     F_matrix[2*i, 2*i + 1] = dt
# print(F_matrix)
# def consolidate_same_measure_distance(measure_index_list, measure_distance_list):
#     # measure_index_list = [[model1_index],[model2_index],....]
#     measure_index_temp = measure_index_list[0]
#     measure_distance_temp = measure_distance_list[0]
#     for i in range(1,len(measure_index_list)):
#         for j in range(len(measure_index_list[i])):
#             if measure_index_list[i][j] not in measure_index_temp:
#                 measure_index_temp.append(measure_index_list[i][j])
#                 measure_distance_temp.append(measure_distance_list[i][j])
#             else:
#                 indx = measure_index_temp.index(measure_index_list[i][j])
#                 measure_distance_temp[indx] = measure_distance_temp[indx] if measure_distance_temp[indx] <= \
#                                                                              measure_distance_list[i][j] \
#                     else measure_distance_list[i][j]
#     return measure_index_temp, measure_distance_temp
def consolidate_same_measure_distance(measure_index_list, measure_distance_list):
    # measure_index_list = [[model1_index],[model2_index],....]
    measure_index_temp = measure_index_list[0]
    measure_distance_temp = measure_distance_list[0]
    model_ind = [0 for _ in range(len(measure_index_list[0]))]
    for i in range(1, len(measure_index_list)):
        for j in range(len(measure_index_list[i])):
            if measure_index_list[i][j] not in measure_index_temp:
                measure_index_temp.append(measure_index_list[i][j])
                measure_distance_temp.append(measure_distance_list[i][j])
                model_ind.append(i)
            else:
                indx = measure_index_temp.index(measure_index_list[i][j])
                model_ind[indx] = model_ind[indx] if measure_distance_temp[indx] <= measure_distance_list[i][j] \
                    else i
                measure_distance_temp[indx] = measure_distance_temp[indx] if measure_distance_temp[indx] <= \
                                                                             measure_distance_list[i][j] \
                    else measure_distance_list[i][j]

    return measure_index_temp, measure_distance_temp, model_ind
a = [[1,2,3,4],
     [3,4,5,6],
     [4,5,6,7]]
dis = [[1.1,1.2,1.3,1.4],
       [1.2,1.5,1.7,1.8],
       [1.1,1.9,1.4,1.6]]
# [1.1,1.2,1.2,1.1,1.7,1.4,1.6]
a,dis,model_ind = consolidate_same_measure_distance(a,dis)
print(a)