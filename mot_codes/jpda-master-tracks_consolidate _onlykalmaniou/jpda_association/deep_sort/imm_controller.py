import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

from .dynamic_model import  Dynamic_model
import scipy.linalg
from jpda_association.application_util.math_operation import MathOperation
#按自由度排布的卡方分布0.95分位数，用于门控距离阈值的选择
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}


class Imm_Controller:
    '''
    IMM_Controller
    每个轨迹(目标)有一个
    k-1时刻,当进行k时刻的位置更新时,IMM根据各个动态模型的概率和位置计算目标的均值位置和协方差
    所以k-1时刻位置的均值和协方差应该为每个动态模型所有
    '''
    def __init__(self, dynamic_model_list=['constant_velocity'],\
                 model_max_velocity=[2,2,0,0],\
                 deviation_process_noise = 0.5,\
                 deviation_measurement_noise=5,\
                 dynamic_model_transition_pro_matrix=[[1.0]],\
                 Gate_value=[(30)**0.5],
                 ndim = 4,
                 dt = 1.):
        # 动态模型概率转移矩阵
        self.dynamic_model_transition_pro_matrix = dynamic_model_transition_pro_matrix
        # 动态模型概率列表，存储每个模型的概率
        self.dynamic_model_prob = None
        # dynamic_model_list 为字符串列表，constant velocity,  constant accelerate, turn around, backforward
        # 动态模型列表('constant_velocity','cv','')
        self.dynamic_model_list = dynamic_model_list
        # 动态模型最大速度
        self.model_max_velocity = model_max_velocity
        # 动态模型数目，作用是初始化时分配模型概率
        self.dynamic_model_num = len(dynamic_model_list)
        # constant model 初始化概率为0.8
        # 未分配概率，作用是初始化时分配模型概率
        self.on_distribute_probability = 1
        # 未分配概率的模型数，作用是初始化时分配模型概率
        self.on_distribute_model = len(dynamic_model_list)
        # dynamic_models的列表
        self.model_list = []
        # self.model_transmiss_prob_matrix = model_transmiss_prob_matrix
        # q1 = 0.5; %The standard deviation of the process noise for the dynamic model
        self.Gate_value = Gate_value
        self.deviation_process_noise = 0.5
        # qm = 7; %The standard deviation of the measurement noise
        self.deviation_measurement_noise = 5
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160
        self.ndim = ndim
        self.dt = dt
        self.math_operation = MathOperation()




    def initiate_dynamic_model(self,measurement,dmean):
        """
        '''
        根据目标框检测值初始化轨迹
        measurement：目标框测量向量(x, y, a, h)，中心点横纵坐标x, y，宽高比a，高h
        '''
        mean_pos = measurement #位置状态分布向量(均值)，维度(4, )
        mean_vel = np.zeros_like(mean_pos) #速度状态分布向量(均值)，维度(4, )
        mean = np.r_[mean_pos, mean_vel] #位置、速度状态分布向量(均值)，维度(8×1)
        #位置、速度状态分布值(标准差)，维度(8, )
        covariance = np.diag(np.square(std)) #位置、速度状态分布矩阵(方差)，维度(8×8)
        return mean, covariance
        """
        """Create track from unassociated measurement.
        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.
        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.
        """
        '''
            根据目标框检测值初始化轨迹
            measurement：目标框测量向量(x, y, a, h),中心点横纵坐标x, y,宽高比a,高h
        '''
        # np.r_扩展行。
        # np.c_扩展列。
        # x,y,a,h
        # 如果没有选其他模型,则默认常数模型,否则加入其他模型
        # ndim 目标状态向量维数
        ndim, dt = self.ndim,self.dt
        # 目标位置状态
        mean_pos = np.zeros_like(measurement)
        # 0,0,0,0
        # 目标速度状态
        mean_vel = np.zeros_like(measurement)
        # x,dx,y,dy,a,da,h,dh
        # 目标状态向量
        mean = np.r_[mean_pos, mean_vel]
        for i in range(ndim):
            mean[2*i] = measurement[i]
            mean[2*i+1] = dmean[i]
        """
        mean_pos = measurement #位置状态分布向量(均值)，维度(4, )
        mean_vel = np.zeros_like(mean_pos) #速度状态分布向量(均值)，维度(4, )
        mean = np.r_[mean_pos, mean_vel] #位置、速度状态分布向量(均值)，维度(8×1)
        std = [
            2 * self._std_weight_position * measurement[0],
            2 * self._std_weight_position * measurement[1],
            1 * measurement[2],
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[0],
            10 * self._std_weight_velocity * measurement[1],
            0.1 * measurement[2],
            10 * self._std_weight_velocity * measurement[3]]
            #位置、速度状态分布值(标准差)，维度(8, )
        covariance = np.diag(np.square(std)) #位置、速度状态分布矩阵(方差)，维度(8×8)
        """
        """
        # _std_weight_position: 控制位置方差权重
        # _std_weight_velocity: 控制速度方差权重
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160
        """
        # measurement[3]: h
        # 协方差矩阵
        std = [self._std_weight_position * measurement[3],
               self._std_weight_velocity * measurement[3],
               self._std_weight_position * measurement[3],
               self._std_weight_velocity * measurement[3],
               1e-2,
               1e-5,
               self._std_weight_position * measurement[3],
               self._std_weight_velocity * measurement[3]
              ]
        # numpy的square用于计算各元素的平方：
        # 协方差、误差估计
        """
        kalman filter
            # x' = Fx
            # P' = FPF^T+Q
        mean为x'
        covariance为P'
        """
        covariance = np.diag(std)
        # Q11x=q1*[T^3/3 T^2/2;T^2/2 T];
        # Q11y=q1*[T^3/3 T^2/2;T^2/2 T];
        # model.Q(:,:,1)=blkdiag(Q11x,Q11y); % The process covariance matrix for the dynamic model 1
        # % if you have more than one linear motion model, add same as Model 1 as
        # % model.F(:,:,N) and model.Q(:,:,N).
        # % Measurement model
        # model.H=[1 0 0 0;0 0 1 0]; % Measurement matrix
        # model.R=qm*eye(2); % Measurement covariance matrix

        self.dynamic_model_prob = []
        for model_name in self.dynamic_model_list:
            if model_name == 'constant_velocity':
                # 需要提供模型概率ui,F,Q,H,R,X(k-1),P(k-1)
                model_prob = 0.7
                self.on_distribute_probability = self.on_distribute_probability-model_prob
                model_prob = 0.7 + self.on_distribute_probability / self.on_distribute_model
                self.on_distribute_model -=1
                # eye是对角矩阵
                # _motion_mat: 状态转移矩阵F，维度(8×8)
                # Create Kalman filter model matrices.
                # eye是对角矩阵
                # _motion_mat: 状态转移矩阵F，维度(8×8)
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
                for i in range(ndim):
                    F_matrix[2*i, 2*i + 1] = dt
                """
                [[1,1,0,0,0,0,0,0],
                 [0,1,0,0,0,0,0,0],
                 [0,0,1,1,0,0,0,0],
                 [0,0,0,1,0,0,0,0],
                 [0,0,0,0,1,1,0,0],
                 [0,0,0,0,0,1,0,0],
                 [0,0,0,0,0,0,1,1],
                 [0,0,0,0,0,0,0,1]
                ]
                """
                # _update_mat: 状态空间向测量空间转移矩阵H，(维度4×8)
                # _std_weight_position: 控制位置方差权重
                # _std_weight_velocity: 控制速度方差权重
                # Fx0 = [x+dx,dx,y+dy,dy,a+da,da,h+dh,dh]
                H_matrix = np.zeros((ndim, 2 * ndim))
                for i in range(ndim):
                    H_matrix[i, 2*i] = dt
                # [[1, 0, 0, 0, 0, 0, 0, 0],
                #  [0, 0, 1, 0, 0, 0, 0, 0],
                #  [0, 0, 0, 0, 1, 0, 0, 0],
                #  [0, 0, 0, 0, 0, 0, 1, 0]]
                # Motion and observation uncertainty are chosen relative to the current state estimate.
                # These weights control the amount of uncertainty in the model. This is a bit hacky.
                # _std_weight_position: 控制位置方差权重
                # _std_weight_velocity: 控制速度方差权重

                process_noise_matrix_Q = np.zeros((2 * ndim, 2 * ndim))  # 8*8
                Q_positon_x = self.deviation_process_noise * np.array([[1 / 4 * dt ** 4, 1 / 2 * dt ** 3],
                [1 / 2 * dt ** 3, dt ** 2]])
                Q_positon_y = self.deviation_process_noise * np.array([[1 / 4 * dt ** 4, 1 / 2 * dt ** 3],
                [1 / 2 * dt ** 3, dt ** 2]])
                process_noise_matrix_Q[0:2,0:2] = Q_positon_x
                process_noise_matrix_Q[2:4,2:4] = Q_positon_y
                measurement_noise_matrix_R = np.eye(ndim)
                measurement_noise_matrix_R[0, 0] = self.deviation_measurement_noise
                measurement_noise_matrix_R[1, 1] = self.deviation_measurement_noise
                constant_model = Dynamic_model(prob_mui=model_prob, mean_xk0=mean, convariance_xk0=covariance,
                                                    F_matrix=F_matrix, H_matrix=H_matrix,\
                                                     Q_matrix=process_noise_matrix_Q,\
                                                     R_matrix=measurement_noise_matrix_R,\
                                                             model_max_velocity=self.model_max_velocity)
                self.model_list.append(constant_model)
                self.dynamic_model_prob.append(model_prob)
            elif model_name == 'constant_accelerate':
                model_prob = self.on_distribute_probability/self.on_distribute_model
                self.on_distribute_probability = self.on_distribute_probability-model_prob
                self.on_distribute_model -= 1
                constant_accelerate_model = Dynamic_model()
                self.model_list.append(constant_accelerate_model)
                self.dynamic_model_prob.append(model_prob)
            elif model_name == 'turn_around':
                model_prob = self.on_distribute_probability / self.on_distribute_model
                self.on_distribute_probability = self.on_distribute_probability - model_prob
                self.on_distribute_model -= 1
                turn_around_model = Dynamic_model()
                self.model_list.append(turn_around_model)
                self.dynamic_model_prob.append(model_prob)
            elif model_name == 'backforward':
                model_prob = self.on_distribute_probability / self.on_distribute_model
                self.on_distribute_probability = self.on_distribute_probability - model_prob
                self.on_distribute_model -= 1
                backward_model = Dynamic_model()
                self.model_list.append(backward_model)
                self.dynamic_model_prob.append(model_prob)
            else:
                # 当作常数模型处理
                # 需要提供模型概率ui,F,Q,H,R,X(k-1),P(k-1)
                model_prob = 0.7
                self.on_distribute_probability = self.on_distribute_probability - model_prob
                model_prob = 0.7 + self.on_distribute_probability / self.on_distribute_model
                self.on_distribute_model -= 1
                # eye是对角矩阵
                # _motion_mat: 状态转移矩阵F，维度(8×8)
                # Create Kalman filter model matrices.
                # eye是对角矩阵
                # _motion_mat: 状态转移矩阵F，维度(8×8)
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
                """
                for i in range(ndim):
                    F_matrix[i, ndim + i] = dt
                """
                [[1,0,0,0,1,0,0,0],
                 [0,1,0,0,0,1,0,0],
                 [0,0,1,0,0,0,1,0],
                 [0,0,0,1,0,0,0,1],
                 [0,0,0,0,1,0,0,0],
                 [0,0,0,0,0,1,0,0],
                 [0,0,0,0,0,0,1,0],
                 [0,0,0,0,0,0,0,1]
                ]
                """
                # _update_mat: 状态空间向测量空间转移矩阵H，(维度4×8)
                # _std_weight_position: 控制位置方差权重
                # _std_weight_velocity: 控制速度方差权重
                H_matrix = np.zeros((ndim, 2 * ndim))
                for i in range(ndim):
                    H_matrix[i, 2 * i] = dt
                # [[1, 0, 0, 0, 0, 0, 0, 0],
                #  [0, 1, 0, 0, 0, 0, 0, 0],
                #  [0, 0, 1, 0, 0, 0, 0, 0],
                #  [0, 0, 0, 1, 0, 0, 0, 0]]
                # Motion and observation uncertainty are chosen relative to the current state estimate.
                # These weights control the amount of uncertainty in the model. This is a bit hacky.
                # _std_weight_position: 控制位置方差权重
                # _std_weight_velocity: 控制速度方差权重

                process_noise_matrix_Q = np.zeros((2 * ndim, 2 * ndim))  # 8*8
                Q_positon_x = self.deviation_process_noise * np.array([[1 / 4 * dt ** 4, 1 / 2 * dt ** 3],
                [1 / 2 * dt ** 3, dt ** 2]])
                Q_positon_y = self.deviation_process_noise * np.array([[1 / 4 * dt ** 4, 1 / 2 * dt ** 3],
                [1 / 2 * dt ** 3, dt ** 2]])
                process_noise_matrix_Q[0:2,0:2] = Q_positon_x
                process_noise_matrix_Q[2:4,2:4] = Q_positon_y
                measurement_noise_matrix_R = np.eye(ndim)
                measurement_noise_matrix_R[0, 0] = self.deviation_measurement_noise
                measurement_noise_matrix_R[1, 1] = self.deviation_measurement_noise
                constant_model = Dynamic_model(prob_mui=model_prob, mean_xk0=mean,
                                                             convariance_xk0=covariance,
                                                             F_matrix=F_matrix, H_matrix=H_matrix, \
                                                             Q_matrix=process_noise_matrix_Q, \
                                                             R_matrix=measurement_noise_matrix_R, \
                                                             model_max_velocity=self.model_max_velocity)
                self.model_list.append(constant_model)
                self.dynamic_model_prob.append(model_prob)
        self.dynamic_model_prob = np.array(self.dynamic_model_prob)[np.newaxis,:]
        if len(self.dynamic_model_list) == 1:
            self.dynamic_model_transition_pro_matrix = np.array([[1.0]])
        else:
            TPM = np.zeros((len(self.dynamic_model_list),len(self.dynamic_model_list)))
            for model_i in range(len(self.dynamic_model_list)):
                for model_j in range(len(self.dynamic_model_list)):
                    if model_i == model_j:
                        TPM[model_i,model_j] = 0.8
                    else:
                        TPM[model_i,model_j] = 0.2/(len(self.dynamic_model_list)-1)
            self.dynamic_model_transition_pro_matrix = TPM.copy()
            # H_TPM = np.zeros((ndim, 2 * ndim))
            # for i in range(ndim):
            #     H_TPM[i, 2 * i + 1] = 1
            # for model_ind,model_name in enumerate(self.dynamic_model_list):

        return mean,covariance

    def predict(self):
        # 第一步，计算c, self.dynamic_model_prob(1,n), self.dynamic_model_transition_pro_matrix(n,n)
        #model_C_matrix = [[n->1, n->2, ...]]
        model_C_matrix = np.dot(self.dynamic_model_prob, self.dynamic_model_transition_pro_matrix)
        # 计算uij
        Uij_matrix = np.zeros_like(self.dynamic_model_transition_pro_matrix)
        for i in range(len(self.dynamic_model_list)):
            for j in range(len(self.dynamic_model_list)):
                Uij_matrix[i,j] = self.dynamic_model_prob[0,i]*self.dynamic_model_transition_pro_matrix[i,j]/model_C_matrix[0,j]
        #计算x0j,计算p0j, X_each_model_matrix,每个模型X（k-1|k-1）
        X0j_matrix = np.zeros((len(model_C_matrix),1,2*self.ndim))
        P0j_matrix = np.zeros((len(model_C_matrix),2*self.ndim,2*self.ndim))
        for model_i in range(len(self.dynamic_model_list)):
            for model_j in range(len(self.dynamic_model_list)):
                X0j_matrix[model_i,:,:] = X0j_matrix[model_i,:,:] + self.model_list[model_j].get_mean_xk0()*Uij_matrix[model_j,model_i]

        #X0j_matrix = [X01,X02,X03,...]
        # X0j_matrix = np.dot(X_each_model_matrix, Uij_matrix)
        # P0j_matrix = np.zeros_like(X0j_matrix)
        for model_i in range(len(self.dynamic_model_list)):
            for model_j in range(len(self.dynamic_model_list)):
                    P0j_matrix[model_i,:,:] = P0j_matrix[model_i,:,:] + Uij_matrix[model_j,model_i]*\
                                           (self.model_list[model_j].get_convariance_xk0() + \
                                            np.dot((self.model_list[model_j].get_mean_xk0() - X0j_matrix[model_i,:,:]),\
                                                   (self.model_list[model_j].get_mean_xk0() - X0j_matrix[model_i,:,:]).T))
        self.Uij_matrix = Uij_matrix
        self.model_C_matrix = model_C_matrix
        self.X0j_matrix = X0j_matrix
        self.P0j_matrix = P0j_matrix

    def kalman_filter_predict(self):
        X_kalman_filter_matrix = np.zeros((len(self.model_list),1,2*self.ndim))
        # 将均值向量映射到检测空间，即fx'
        for i in range(len(self.model_list)):
            X_kalman_filter_matrix[i,:,:] = np.dot(self.model_list[i].get_F_matrix(),\
                                                         self.X0j_matrix[i,:,:].T).T
            self.model_list[i].set_kalman_filter_X0(X_kalman_filter_matrix[i,:,:])
        P_kalman_filter_matrix = np.zeros((len(self.model_list),2*self.ndim,2*self.ndim))
        # 将协方差矩阵映射到检测空间，即HP'H^T
        # covariance + innovation_cov: HP'H^T + R
        for i in range(len(self.model_list)):
            P_kalman_filter_matrix[i,:,:] = np.linalg.multi_dot((
            self.model_list[i].get_F_matrix(), self.P0j_matrix[i,:,:], self.model_list[i].get_F_matrix().T))+\
                                          self.model_list[i].get_Q_matrix()
            self.model_list[i].set_kalman_filter_P0(P_kalman_filter_matrix[i,:,:])
        self.X_kalman_filter_matrix = X_kalman_filter_matrix
        self.P_kalman_filter_matrix = P_kalman_filter_matrix

    def pdist(self,a, b, threshold):
        # 译文：计算“a”和“b”点之间的成对距离的平方
        """Compute pair-wise squared distance between points in `a` and `b`.

        Parameters
        ----------
        a : array_like
            An NxM matrix of N samples of dimensionality M.
        b : array_like
            An LxM matrix of L samples of dimensionality M.

        Returns
        -------
        ndarray
            Returns a matrix of size len(a), len(b) such that eleement (i, j)
            contains the squared distance between `a[i]` and `b[j]`.

        """
        """
        '''
        # 用于计算成对的平方距离
        # a NxM 代表N个对象，每个对象有M个数值作为embedding进行比较
        # b LxM 代表L个对象，每个对象有M个数值作为embedding进行比较
        # 返回的是NxL的矩阵，比如dist[i][j]代表a[i]和b[j]之间的平方和距离
        # 实现见：https://blog.csdn.net/frankzd/article/details/80251042
        '''
        格式：np.sum(a)
        np.sum(a, axis=0) ------->列求和
        np.sum(a, axis=1) ------->行求和
        """
        # 拷贝一份数据
        a, b = np.asarray(a), np.asarray(b)
        if len(a) == 0 or len(b) == 0:
            # (N,0) or (0,L)
            return np.zeros((len(a), len(b)))
        # a: NM; b: LM
        # a2:N ; b2:L
        a2, b2 = np.square(a).sum(axis=1), np.square(b).sum(axis=1)
        # np.dot(a,b.T): L=NL
        # a2[:,None]: N1   b2[None:]: 1L
        # (a[i]-b[i])**2
        # 求每个embedding的平方和
        # sum(N) + sum(L) -2 x [NxM]x[MxL] = [NxL]
        r2 = -2. * np.dot(a, b.T) + a2[:, None] + b2[None, :]
        # np.inf :+oo
        # r2 = np.clip(r2, 0., threshold)
        return r2



    def _nn_cosine_distance(self,x,y,threshold):
        """ Helper function for nearest neighbor distance metric (Euclidean).

               Parameters
               ----------
               x : ndarray
                   A matrix of N row-vectors (sample points).
               y : ndarray
                   A matrix of M row-vectors (query points).

               Returns
               -------
               ndarray
                   A vector of length M that contains for each entry in `y` the
                   smallest Euclidean distance to a sample in `x`.

               """
        y = [det.to_xyah()[:2] for det in y]
        distances = self.pdist(x[:,0:2], y, threshold)
        detection_index = []
        distance_detection_and_target = []
        for detection_i in range(distances.shape[1]):
            if distances[0, detection_i] < threshold:
                detection_index.append(detection_i)
                distance_detection_and_target.append(distances[0, detection_i])
        return detection_index, distance_detection_and_target

    def _nn_cosine_sita_distance(self,x,y,sita,threshold):
        """ Helper function for nearest neighbor distance metric (Euclidean).

               Parameters
               ----------
               x : ndarray
                   A matrix of N row-vectors (sample points).
               y : ndarray
                   A matrix of M row-vectors (query points).

               Returns
               -------
               ndarray
                   A vector of length M that contains for each entry in `y` the
                   smallest Euclidean distance to a sample in `x`.

               """
        y = [det.to_bcxywh()[:2] for det in y]
        trackbxy = x[:,0:2]
        trackbxy[:,1] += x[:,3]/2
        distances = self.pdist(trackbxy, y, threshold)
        detection_index = []
        distance_detection_and_target = []
        threshold = float(threshold)
        for detection_i in range(distances.shape[1]):
            if distances[0, detection_i] < threshold:
                # dsita = self.math_operation.get_track_direction_sita(y[detection_i],trackbxy[0])
                # if np.abs(dsita - sita) <= (1/3)*np.pi:
                detection_index.append(detection_i)
                distance_detection_and_target.append(distances[0, detection_i])
        return detection_index, distance_detection_and_target


    def _nn_Mahalanobis_distance(self, mean, covariance, measurements,
                                 only_position=False):
        # 计算状态分布和测量之间的门控距离。
        # 这里因为位置变为了（x,dx,y,dy,...），所以如果只要position，需要改
        gating_dim = 2 if only_position else 4
        measurements = np.array([det.to_xyah() for det in measurements])
        mean = np.array([mean[0] for _ in measurements])
        if only_position:
            mean = mean[:2]
            covariance = covariance[:2, :2]
            measurements = measurements[:, :2]
        gating_threshold = chi2inv95[gating_dim]
        cholesky_factor = np.linalg.cholesky(covariance)
        # 计算的是Detection和Track的均值误差。
        d = measurements - mean
        # 解方程，得到choleskey_factor x = d.T 的x解
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False,
            overwrite_b=True)
        squared_maha = np.sum(z*z, axis=0)
        detection_index = []
        distance_detection_and_target = []
        for detection_i in range(len(squared_maha)):
            if squared_maha[detection_i] <= gating_threshold:
                detection_index.append(detection_i)
                distance_detection_and_target.append(squared_maha[detection_i])
        return detection_index, distance_detection_and_target

    def iou(self,bbox, candidates):
        """Computer intersection over union.

        Parameters
        ----------
        bbox : ndarray
            A bounding box in format `(top left x, top left y, width, height)`.
        candidates : ndarray
            A matrix of candidate bounding boxes (one per row) in the same format
            as `bbox`.

        Returns
        -------
        ndarray
        在[0,1]中，“bbox”和每个候选点之间的交集。分数越高，意味着候选人遮挡的“b盒”的比例越大。
            The intersection over union in [0, 1] between the `bbox` and each
            candidate. A higher score means a larger fraction of the `bbox` is
            occluded by the candidate.

        """
        bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
        candidates_tl = candidates[:, :2]
        candidates_br = candidates[:, :2] + candidates[:, 2:]

        tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
        np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
        br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
        np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
        wh = np.maximum(0., br - tl)

        area_intersection = wh.prod(axis=1)
        area_bbox = bbox[2:].prod()
        area_candidates = candidates[:, 2:].prod(axis=1)
        return area_intersection / (area_bbox + area_candidates - area_intersection)

    def _nn_iou_distance(self,mean,x,y,sita,threshold):
        cx,cy,a,h = x[0]
        # ret[:2] -= ret[2:] / 2

        bbox = np.array([cx-0.5*a*h,cy-0.5*h,a*h,h])
        # bbox = tracks[track_idx].to_tlwh()
        candidates = np.asarray([i.tlwh for i in y])
        distances = self.iou(bbox, candidates)
        y = [det.to_bcxywh()[:2] for det in y]
        # mean : x,dx,y,dy,a,da,h,dh
        trackbxy = mean[0:2]
        trackbxy[1] = mean[2] + mean[6]/2
        # distances = self.pdist(trackbxy, y, threshold)
        detection_index = []
        distance_detection_and_target = []
        for detection_i in range(len(distances)):
            if distances[detection_i] > threshold:
                dsita = self.math_operation.get_track_direction_sita(y[detection_i],trackbxy)
                if np.abs(dsita - sita) <= 180:
                    detection_index.append(detection_i)
                    distance_detection_and_target.append(distances[detection_i])
        return detection_index, distance_detection_and_target

    def find_detections_in_association_Doors_by_iou(self,detections,sita,frame_idx):
        measure_index_list = []
        measure_distance_list = []
        for i,model in enumerate(self.model_list):
            # 将均值向量映射到检测空间，即Hx'
            mean_t = np.dot(model.get_kalman_filter_X0(),model.get_H_matrix().T)
            # 将协方差矩阵映射到检测空间，即HP'H^T
            covariance_t = np.linalg.multi_dot((
                model.get_H_matrix(), model.get_kalman_filter_P0(), model.get_H_matrix().T))+model.get_R_matrix()
            # covariance + innovation_cov: HP'H^T + R
            distance_threshold = 0.4
            measure_index,measure_distance = self._nn_iou_distance(model.get_mean_xk0(),\
                mean_t,detections,sita,distance_threshold)

            # measure_index,measure_distance = self._nn_cosine_distance(\
            #     mean_t,detections,self.Gate_value)
            # measure_index,measure_distance = self._nn_cosine_sita_distance(\
            #     mean_t,detections,sita,distance_threshold)
            # measure_index, measure_distance = self._nn_Mahalanobis_distance( \
            #     mean_t, covariance_t, detections)
            measure_index_list.append(measure_index)
            measure_distance_list.append(measure_distance)
            model.set_possible_measure_index(measure_index)
            model.set_possible_measure_distance(measure_distance)
        return measure_index_list,measure_distance_list

    def find_detections_in_association_Doors(self,detections,sita,frame_idx):
        measure_index_list = []
        measure_distance_list = []
        for i,model in enumerate(self.model_list):
            # 将均值向量映射到检测空间，即Hx'
            mean_t = np.dot(model.get_kalman_filter_X0(),model.get_H_matrix().T)
            # 将协方差矩阵映射到检测空间，即HP'H^T
            covariance_t = np.linalg.multi_dot((
                model.get_H_matrix(), model.get_kalman_filter_P0(), model.get_H_matrix().T))+model.get_R_matrix()
            # covariance + innovation_cov: HP'H^T + R
            distance_threshold = 256**(0.5)
            # measure_index,measure_distance = self._nn_cosine_distance(\
            #     mean_t,detections,self.Gate_value)
            measure_index,measure_distance = self._nn_cosine_sita_distance(\
                mean_t,detections,sita,distance_threshold)
            # measure_index, measure_distance = self._nn_Mahalanobis_distance( \
            #     mean_t, covariance_t, detections)
            measure_index_list.append(measure_index)
            measure_distance_list.append(measure_distance)
            model.set_possible_measure_index(measure_index)
            model.set_possible_measure_distance(measure_distance)
        return measure_index_list,measure_distance_list

    def cal_total_mean_and_covariance(self):
        mean = np.zeros_like(self.model_list[0].get_prob_mui()*self.model_list[0].get_update_mean_X())
        convariance = np.zeros_like(self.model_list[0].get_prob_mui()* self.model_list[0].get_update_covariance_P())
        for model_i in range(len(self.model_list)):
            mean += self.model_list[model_i].get_prob_mui()*self.model_list[model_i].get_update_mean_X()
        for model_j in range(len(self.model_list)):
            convariance += self.model_list[model_j].get_prob_mui()*\
                           (self.model_list[model_j].get_update_covariance_P()+\
                            np.dot((self.model_list[model_j].get_update_mean_X()-mean).T,\
                                   (self.model_list[model_j].get_update_mean_X()-mean)))
        return mean[0], convariance



    def update_by_jpda(self,jpda_prob_matrix, detections, track_idx,frame_idx):
        for model_i in range(len(self.model_list)):
            # 将均值和协方差映射到检测空间，得到 Hx' 和 S
            # fx
            kalman_filter_X0_fx = self.model_list[model_i].get_kalman_filter_X0()
            # fpft+Q
            kalman_filter_P0_fpf_Q = self.model_list[model_i].get_kalman_filter_P0()
            # R是目标检测器的噪声矩阵，是一个4x4的对角矩阵。 对角线上的值分别为中心点两个坐标以及宽高的噪声。
            innovation_cov_R = self.model_list[model_i].get_R_matrix()
            update_mat_H = self.model_list[model_i].get_H_matrix()

            # 将均值向量映射到检测空间，即Hx'
            # hx
            mean_hx = np.dot(kalman_filter_X0_fx,update_mat_H.T)
            # 将协方差矩阵映射到检测空间，即HP'H^T
            covariance_hph = np.linalg.multi_dot((
                update_mat_H, kalman_filter_P0_fpf_Q, update_mat_H.T))
            # covariance + innovation_cov: HP'H^T + R
            # hph+R
            covariance_hph_R = covariance_hph + innovation_cov_R
            # # 矩阵分解
            # cholesky分解是一种将任意n阶对称正定矩阵A分解成下三角矩阵L的一种方法：
            chol_factor, lower = scipy.linalg.cho_factor(
                covariance_hph_R, lower=True, check_finite=False)
            # 计算卡尔曼增益K K=P′HTS −1
            # np.dot(covariance, self._update_mat.T)： Sx ,(P'H.T).T = HP'.T
            # return  (S-1HP'.T).T   =  P'H.T S-1
            kalman_gain = scipy.linalg.cho_solve(
                (chol_factor, lower), np.dot(kalman_filter_P0_fpf_Q, update_mat_H.T).T,
                check_finite=False).T
            """
            这个公式中，z是Detection的mean，不包含变化值，
            状态为[cx,cy,a,h]。H是测量矩阵，将Track的均值向量x ′ x'x ′
            映射到检测空间。计算的y是Detection和Track的均值误差。
            """
            # y = z−Hx′
            detection_indexs = self.model_list[model_i].get_possible_measure_index()
            distance_detection_and_track = self.model_list[model_i].get_possible_measure_distance()
            # 这里是为了获得np.dot(dv, prob)的形状并初始化
            prob_dv0 = None
            for detections_id in detection_indexs:
                dv0 = detections[detections_id].to_xyah()-mean_hx
                prob0 = jpda_prob_matrix[detections_id+1,track_idx]
                prob_dv0 = np.zeros_like(np.dot(dv0, prob0))
                break
            for detections_id in detection_indexs:
                dv0 = detections[detections_id].to_xyah() - mean_hx
                prob0 = jpda_prob_matrix[detections_id+1, track_idx]
                prob_dv0 += np.dot(dv0, prob0)
            # x = x' + Ky
            # new_mean = mean + np.dot(innovation, kalman_gain.T)
            mean_update = kalman_filter_X0_fx + np.dot(prob_dv0, kalman_gain.T)
            # P = (I - KH)P'
            # new_covariance = covariance - np.linalg.multi_dot((
            #     kalman_gain, projected_cov, kalman_gain.T))
            beta0t = jpda_prob_matrix[0, track_idx]
            covariance_update_ksk = np.linalg.multi_dot((kalman_gain,covariance_hph_R,kalman_gain.T))

            kdvk_dv = np.zeros((self.ndim,self.ndim))
            for detections_id in detection_indexs:
                dv1 = detections[detections_id].to_xyah() - mean_hx
                prob1 = jpda_prob_matrix[detections_id+1, track_idx]
                kdvk_dv += prob1*np.dot(dv1.T, dv1)-np.dot(prob_dv0.T,prob_dv0)
            kdvk = np.linalg.multi_dot((kalman_gain,kdvk_dv.T,kalman_gain.T))
            covariance_update = kalman_filter_P0_fpf_Q - (1-beta0t)*covariance_update_ksk+kdvk

            mean_dim = update_mat_H.shape[0]
            dv_s_dv = np.linalg.multi_dot((prob_dv0, np.linalg.inv(covariance_hph_R), prob_dv0.T))
            model_prob_estimate = (1/(((2*np.pi)**(mean_dim/2))*(np.linalg.det(covariance_hph_R)**(0.5))))*np.exp(-0.5*dv_s_dv)

            self.model_list[model_i].set_update_mean_X(mean_update)
            self.model_list[model_i].set_update_covariance_P(covariance_update)
            self.model_list[model_i].set_model_prob_estimate(model_prob_estimate)
            self.model_list[model_i].set_prob_mui(model_prob_estimate)


        #     此时已经得到每个模型的update_mean、update_covariance和模型概率似然估计
        # 第一步，计算c, self.dynamic_model_prob(1,n), self.dynamic_model_transition_pro_matrix(n,n)
        # model_C_matrix = [[n->1, n->2, ...]]
        model_C_matrix = np.dot(self.dynamic_model_prob, self.dynamic_model_transition_pro_matrix)
        normalized_c_of_model_prob_estimate = 0.0
        for model_i in range(len(self.model_list)):
            normalized_c_of_model_prob_estimate = normalized_c_of_model_prob_estimate+\
                                                  model_C_matrix[0,model_i]*self.model_list[model_i].get_model_prob_estimate()
        normalized_c_of_model_prob_estimate = 1.0 if normalized_c_of_model_prob_estimate == 0.0 else normalized_c_of_model_prob_estimate
        for a_model_i in range(len(self.model_list)):
            self.model_list[a_model_i].set_prob_mui(\
                model_C_matrix[0,a_model_i]*self.model_list[a_model_i].get_model_prob_estimate()/normalized_c_of_model_prob_estimate)
        mean,covariance = self.cal_total_mean_and_covariance()
        return mean,covariance

    def update_by_kalman(self, detection):
        for model_i in range(len(self.model_list)):
            # 将均值和协方差映射到检测空间，得到 Hx' 和 S
            # fx
            # fx
            kalman_filter_X0_fx = self.model_list[model_i].get_kalman_filter_X0()
            # fpft+Q
            kalman_filter_P0_fpf_Q = self.model_list[model_i].get_kalman_filter_P0()
            # R是目标检测器的噪声矩阵，是一个4x4的对角矩阵。 对角线上的值分别为中心点两个坐标以及宽高的噪声。
            innovation_cov_R = self.model_list[model_i].get_R_matrix()
            update_mat_H = self.model_list[model_i].get_H_matrix()

            # 将均值向量映射到检测空间，即Hx'
            # hx
            mean_hx = np.dot(kalman_filter_X0_fx, update_mat_H.T)
            # 将协方差矩阵映射到检测空间，即HP'H^T
            covariance_hph = np.linalg.multi_dot((
                update_mat_H, kalman_filter_P0_fpf_Q, update_mat_H.T))
            # covariance + innovation_cov: HP'H^T + R
            # hph+R
            covariance_hph_R = covariance_hph + innovation_cov_R
            # # 矩阵分解
            # cholesky分解是一种将任意n阶对称正定矩阵A分解成下三角矩阵L的一种方法：
            chol_factor, lower = scipy.linalg.cho_factor(
                covariance_hph_R, lower=True, check_finite=False)
            # 计算卡尔曼增益K K=P′HTS −1
            # np.dot(covariance, self._update_mat.T)： Sx ,(P'H.T).T = HP'.T
            # return  (S-1HP'.T).T   =  P'H.T S-1
            kalman_gain = scipy.linalg.cho_solve(
                (chol_factor, lower), np.dot(kalman_filter_P0_fpf_Q, update_mat_H.T).T,
                check_finite=False).T
            """
            这个公式中，z是Detection的mean，不包含变化值，
            状态为[cx,cy,a,h]。H是测量矩阵，将Track的均值向量x ′ x'x ′
            映射到检测空间。计算的y是Detection和Track的均值误差。
            """
            # y = z−Hx′
            detection_indexs = self.model_list[model_i].get_possible_measure_index()
            distance_detection_and_track = self.model_list[model_i].get_possible_measure_distance()

            # 这里是为了获得np.dot(dv, prob)的形状并初始化
            detection_xyah = detection.to_xyah()
            dv0 = detection_xyah - mean_hx

            # # x = x' + Ky
            # new_mean = mean + np.dot(innovation, kalman_gain.T)
            mean_update = kalman_filter_X0_fx + np.dot(dv0, kalman_gain.T)
            # P = (I - KH)P'
            # new_covariance = covariance - np.linalg.multi_dot((
            #     kalman_gain, projected_cov, kalman_gain.T))
            covariance_update = kalman_filter_P0_fpf_Q-\
                                np.linalg.multi_dot((kalman_gain,covariance_hph_R,kalman_gain.T))


            # dv_s_dv = np.linalg.multi_dot((prob_dv0, np.linalg.inv(covariance_hph_R), prob_dv0.T))
            model_prob_estimate = multivariate_normal.pdf(dv0, mean=np.zeros(self.ndim), cov=covariance_hph_R)
            # model_prob_estimate = (1 / (
            #             ((2 * np.pi) ** (mean_dim / 2)) * (np.linalg.det(covariance_hph_R) ** (0.5)))) * np.exp(
            #     -0.5 * dv_s_dv)

            self.model_list[model_i].set_update_mean_X(mean_update)
            self.model_list[model_i].set_update_covariance_P(covariance_update)
            self.model_list[model_i].set_model_prob_estimate(model_prob_estimate)
            self.model_list[model_i].set_prob_mui(model_prob_estimate)
        #     此时已经得到每个模型的update_mean、update_covariance和模型概率似然估计
        # 第一步，计算c, self.dynamic_model_prob(1,n), self.dynamic_model_transition_pro_matrix(n,n)
        # model_C_matrix = [[n->1, n->2, ...]]
        model_C_matrix = np.dot(self.dynamic_model_prob, self.dynamic_model_transition_pro_matrix)
        normalized_c_of_model_prob_estimate = 0.0
        for model_i in range(len(self.model_list)):
            normalized_c_of_model_prob_estimate = normalized_c_of_model_prob_estimate + \
                                                  model_C_matrix[0, model_i] * self.model_list[
                                                      model_i].get_model_prob_estimate()
        if normalized_c_of_model_prob_estimate != 0.0:
            for a_model_i in range(len(self.model_list)):
                self.model_list[a_model_i].set_prob_mui( \
                    model_C_matrix[0, a_model_i] * self.model_list[
                        a_model_i].get_model_prob_estimate() / normalized_c_of_model_prob_estimate)
        else:
            for a_model_i in range(len(self.model_list)):
                self.model_list[a_model_i].set_prob_mui(0.5)
        mean, covariance = self.cal_total_mean_and_covariance()
        return mean, covariance

    # def initiate(self, measurement):
    #     """
    #     '''
    #     根据目标框检测值初始化轨迹
    #     measurement：目标框测量向量(x, y, a, h)，中心点横纵坐标x, y，宽高比a，高h
    #     '''
    #     mean_pos = measurement #位置状态分布向量(均值)，维度(4, )
    #     mean_vel = np.zeros_like(mean_pos) #速度状态分布向量(均值)，维度(4, )
    #     mean = np.r_[mean_pos, mean_vel] #位置、速度状态分布向量(均值)，维度(8×1)
    #
    #         #位置、速度状态分布值(标准差)，维度(8, )
    #     covariance = np.diag(np.square(std)) #位置、速度状态分布矩阵(方差)，维度(8×8)
    #     return mean, covariance
    #     """
    #     """Create track from unassociated measurement.
    #
    #     Parameters
    #     ----------
    #     measurement : ndarray
    #         Bounding box coordinates (x, y, a, h) with center position (x, y),
    #         aspect ratio a, and height h.
    #
    #     Returns
    #     -------
    #     (ndarray, ndarray)
    #         Returns the mean vector (8 dimensional) and covariance matrix (8x8
    #         dimensional) of the new track. Unobserved velocities are initialized
    #         to 0 mean.
    #
    #     """
    #     '''
    #         根据目标框检测值初始化轨迹
    #         measurement：目标框测量向量(x, y, a, h),中心点横纵坐标x, y,宽高比a,高h
    #     '''
    #     # np.r_是按列连接两个矩阵，就是把两矩阵上下相加，要求列数相等。
    #     # np.c_是按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等。
    #     # x,y,a,h
    #     # 如果没有选其他模型,则默认常数模型,否则加入其他模型
    #
    #
    #     mean_pos = measurement
    #     # 0,0,0,0
    #     mean_vel = np.zeros_like(mean_pos)
    #     mean = np.r_[mean_pos, mean_vel]
    #     """
    #     mean_pos = measurement #位置状态分布向量(均值)，维度(4, )
    #     mean_vel = np.zeros_like(mean_pos) #速度状态分布向量(均值)，维度(4, )
    #     mean = np.r_[mean_pos, mean_vel] #位置、速度状态分布向量(均值)，维度(8×1)
    #     std = [
    #         2 * self._std_weight_position * measurement[0],
    #         2 * self._std_weight_position * measurement[1],
    #         1 * measurement[2],
    #         2 * self._std_weight_position * measurement[3],
    #         10 * self._std_weight_velocity * measurement[0],
    #         10 * self._std_weight_velocity * measurement[1],
    #         0.1 * measurement[2],
    #         10 * self._std_weight_velocity * measurement[3]]
    #         #位置、速度状态分布值(标准差)，维度(8, )
    #     covariance = np.diag(np.square(std)) #位置、速度状态分布矩阵(方差)，维度(8×8)
    #     """
    #     """
    #     # _std_weight_position: 控制位置方差权重
    #     # _std_weight_velocity: 控制速度方差权重
    #     self._std_weight_position = 1. / 20
    #     self._std_weight_velocity = 1. / 160
    #     """
    #
    #     std = [2 * self._std_weight_position * measurement[3],
    #         2 * self._std_weight_position * measurement[3],
    #         1e-2,
    #         2 * self._std_weight_position * measurement[3],
    #         10 * self._std_weight_velocity * measurement[3],
    #         10 * self._std_weight_velocity * measurement[3],
    #         1e-5,
    #         10 * self._std_weight_velocity * measurement[3]
    #     ]
    #     # numpy的square用于计算各元素的平方：
    #     # 协方差、误差估计
    #
    #     """
    #     kalman filter
    #         # x' = Fx
    #         # P' = FPF^T+Q
    #     mean为x'
    #     covariance为P'
    #     """
    #     covariance = np.diag(np.square(std))
    #     # 用作初始化的模型概率
    #     prob_mui = 1
    #     # eye是对角矩阵
    #     # _motion_mat: 状态转移矩阵F，维度(8×8)
    #     ndim, dt = 4, 1.
    #     # Create Kalman filter model matrices.
    #     # eye是对角矩阵
    #     # _motion_mat: 状态转移矩阵F，维度(8×8)
    #     F_matrix = np.eye(2 * ndim, 2 * ndim)
    #     """
    #     [[1,0,0,0,0,0,0,0],
    #      [0,1,0,0,0,0,0,0],
    #      [0,0,1,0,0,0,0,0],
    #      [0,0,0,1,0,0,0,0],
    #      [0,0,0,0,1,0,0,0],
    #      [0,0,0,0,0,1,0,0],
    #      [0,0,0,0,0,0,1,0],
    #      [0,0,0,0,0,0,0,1]
    #     ]
    #     """
    #     for i in range(ndim):
    #         F_matrix[i, ndim + i] = dt
    #     """
    #     [[1,0,0,0,1,0,0,0],
    #      [0,1,0,0,0,1,0,0],
    #      [0,0,1,0,0,0,1,0],
    #      [0,0,0,1,0,0,0,1],
    #      [0,0,0,0,1,0,0,0],
    #      [0,0,0,0,0,1,0,0],
    #      [0,0,0,0,0,0,1,0],
    #      [0,0,0,0,0,0,0,1]
    #     ]
    #     """
    #     # _update_mat: 状态空间向测量空间转移矩阵H，(维度4×8)
    #     # _std_weight_position: 控制位置方差权重
    #     # _std_weight_velocity: 控制速度方差权重
    #     H_matrix = np.eye(ndim, 2 * ndim)
    #     # [[1, 0, 0, 0, 0, 0, 0, 0],
    #     #  [0, 1, 0, 0, 0, 0, 0, 0],
    #     #  [0, 0, 1, 0, 0, 0, 0, 0],
    #     #  [0, 0, 0, 1, 0, 0, 0, 0]]
    #     # Motion and observation uncertainty are chosen relative to the current state estimate.
    #     # These weights control the amount of uncertainty in the model. This is a bit hacky.
    #     # _std_weight_position: 控制位置方差权重
    #     # _std_weight_velocity: 控制速度方差权重
    #     import dynamic_model
    #     amodel = dynamic_model.DynamicModel(prob_mui=prob_mui, mean_xk0=mean, convariance_xk0=covariance,F_matrix=F_matrix,H_matrix=H_matrix,Q,R)
    #     for model_id in self.model_choices:
    #         if model_id == 1:
    #         # 匀加速模式
    #         elif model_id == 2:
    #         # 转弯模型
    #         elif model_id == 3:
    #         # 倒退模型
    #         else:
    #             # ...静止模型
    #             break
    #     return mean, covariance

    def cal_X0j_P0j(self,measurement):
        return


