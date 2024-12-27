# vim: expandtab:ts=4:sw=4
import numpy as np
import scipy.linalg


"""
quantile n. [计] 分位数；分位点
the chi-square distribution: 卡方分布
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold(马氏门控阈值).
"""
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


class KalmanFilter(object):

    """
    A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space

        x, y, a, h, vx, vy, va, vh
    边界框中心坐标，纵横比，高，
    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities. 各自的速度
    #匀速模型
    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).

    """
    """
        _motion_mat:状态转移矩阵F，维度(8×8)
        _update_mat:状态空间向测量空间转移矩阵H，(维度4×8)
        _std_weight_position:控制位置方差权重
        _std_weight_velocity:控制速度方差权重
        
        X(k) = F(k)X(k-1) + B(k)u(k)
        P(k) = F(k)P(k-1)F(k).T + Q(k)
        X(K):状态向量（均值）
        u: 外部控制向量
        P:状态矩阵（协方差）
        F:状态转移矩阵
        B:外部控制矩阵
        Q:外部干扰矩阵（噪声）
       X(k)' = X(k) + K(z(k) - H(k)X(K)）
       P(k)' = P(k) - KH(k)P(k)
       K = P(k)H(k)T(H(k)P(k)H(k)T +R(k))-1
       X(k)': 最优估计向量（均值）
       z:测量向量（均值）
       H:状态空间向测量空间转移矩阵
       P':最优估计矩阵（协方差）
       R:测量矩阵（协方差）
       K:增益矩阵
    """
    def __init__(self):
        ndim, dt = 4, 1.

        # Create Kalman filter model matrices.
        #eye是对角矩阵
        # _motion_mat: 状态转移矩阵F，维度(8×8)
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
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
            self._motion_mat[i, ndim + i] = dt
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
        # self._update_mat = np.eye(ndim, 2 * ndim)
        self._update_mat = np.zeros((ndim, 2 * ndim))
        for i in range(ndim):
            self._update_mat[i, 2 * i] = dt
        # [[1, 0, 0, 0, 0, 0, 0, 0],
        #  [0, 1, 0, 0, 0, 0, 0, 0],
        #  [0, 0, 1, 0, 0, 0, 0, 0],
        #  [0, 0, 0, 1, 0, 0, 0, 0]]
        # Motion and observation uncertainty are chosen relative to the current state estimate.
        # 相对于当前状态估计选择运动和观测的不确定性
        # These weights control the amount of uncertainty in the model. This is a bit hacky.
        # _std_weight_position: 控制位置方差权重
        # _std_weight_velocity: 控制速度方差权重
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
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
            	measurement：目标框测量向量(x, y, a, h)，中心点横纵坐标x, y，宽高比a，高h
        '''
        # np.r_是按列连接两个矩阵，就是把两矩阵上下相加，要求列数相等。
        # np.c_是按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等。
        # x,y,a,h
        mean_pos = measurement
        # 0,0,0,0
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
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
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]
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
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """Run Kalman filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The 8 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.

        """
        # 相当于得到t时刻估计值
        # Q 预测过程中噪声协方差
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]
                  ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]
                  ]
        # 8*1
        # np.r_ 按列连接两个矩阵
        # 初始化噪声矩阵Q
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        """
        _motion_mat= 
                [[1,0,0,0,1,0,0,0],
                 [0,1,0,0,0,1,0,0],
                 [0,0,1,0,0,0,1,0],
                 [0,0,0,1,0,0,0,1],
                 [0,0,0,0,1,0,0,0],
                 [0,0,0,0,0,1,0,0],
                 [0,0,0,0,0,0,1,0],
                 [0,0,0,0,0,0,0,1]]
        """
        # x' = Fx
        mean = np.dot(self._motion_mat, mean)
        # 就是计算矩阵点乘，可以计算多个矩阵的点乘
        # P' = FPF^T+Q
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean, covariance):
        """Project state distribution to measurement space.

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8 dimensional array).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).

        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.

        """
        # R 测量过程中噪声的协方差
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]]
        # 初始化噪声矩阵R
        # R是目标检测器的噪声矩阵，是一个4x4的对角矩阵。 对角线上的值分别为中心点两个坐标以及宽高的噪声。
        innovation_cov = np.diag(np.square(std))

        # 将均值向量映射到检测空间，即Hx'
        mean = np.dot(self._update_mat, mean)
        # 将协方差矩阵映射到检测空间，即HP'H^T
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        # covariance + innovation_cov: HP'H^T + R
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.

        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.

        """
        # 通过估计值和观测值估计最新结果
        """
        预测估计
        predict()
        x′=Fx
        P′=FPFT+Q
        """
        """
        测量部分（检测）
        project()
        Hx′
        S=HP′HT+R
        R是目标检测器的噪声矩阵，是一个4x4的对角矩阵。 对角线上的值分别为中心点两个坐标以及宽高的噪声。
        卡尔曼增益
        K=P′HTS −1
        计算的是卡尔曼增益，是作用于衡量估计误差的权重。
        最优结果
        x=x′+Ky
        更新后的均值向量x
        P=(I−KH)P′
        更新后的协方差矩阵。
        """

        # 将均值和协方差映射到检测空间，得到 Hx' 和 S
        projected_mean, projected_cov = self.project(mean, covariance)
        # # 矩阵分解
        # cholesky分解是一种将任意n阶对称正定矩阵A分解成下三角矩阵L的一种方法：
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        # 计算卡尔曼增益K K=P′HTS −1
        # np.dot(covariance, self._update_mat.T)： Sx ,(P'H.T).T = HP'.T
        # return  (S-1HP'.T).T   =  P'H.T S-1
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T

        """
        这个公式中，z是Detection的mean，不包含变化值，
        状态为[cx,cy,a,h]。H是测量矩阵，将Track的均值向量x ′ x'x ′
        映射到检测空间。计算的y是Detection和Track的均值误差。
        """
        # y = z−Hx′
        innovation = measurement - projected_mean
        # x = x' + Ky
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        # P = (I - KH)P'
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False):
        # 计算状态分布和测量之间的门控距离。
        """Compute gating distance between state distribution and measurements.

        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.
        从“chi2inv95”可以得到一个合适的距离阈值。如果“only position”为False，
        则卡方分布的自由度为4，否则为2。

        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
            状态分布上的均值向量(8维)。
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
            一个包含N个测量值的Nx4维矩阵，每个测量值的格式为(x, y, a, h)，
            其中(x, y)是边界框的中心位置，a是长宽比，h是高度。
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.
            如果为True，则仅根据边界框中心位置进行距离计算。

        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.
            返回一个长度为N的数组，其中第i个元素包含(均值，协方差)
            和' measurements[i] '之间的马氏距离的平方。
        """
        # 将均值和协方差映射到检测空间，得到 Hx' 和 S
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        cholesky_factor = np.linalg.cholesky(covariance)
        # 计算的是Detection和Track的均值误差。
        d = measurements - mean
        # 解方程，得到choleskey_factor x = d.T 的x解
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False,
            overwrite_b=True)
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha
