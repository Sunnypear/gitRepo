import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.utils.linear_assignment_ import linear_assignment


cost_matrix = np.array([[180., 20., 180., 10., 180., 40.],
                        [30., 180., 70., 80., 180., 20.],
                        [180., 180., 180., 20., 20., 90.],])
row_t, col_t = linear_sum_assignment(cost_matrix, False)
print("s")