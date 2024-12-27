# vim: expandtab:ts=4:sw=4
import numpy as np
from jpda_association.application_util.math_operation import MathOperation

class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """
    # # 单个轨迹的三种状态
    # Tentative = 1  # 不确定态
    # Confirmed = 2  # 确定态
    # Deleted = 3  # 删除态
    Tentative = 1
    Confirmed = 2
    Deleted = 3

    """
    Track类主要存储的是轨迹信息，mean和covariance是保存的框的位置和速度信息，
    track_id代表分配给这个轨迹的ID。
    state代表框的状态，有三种：
    Tentative: 不确定态，这种状态会在初始化一个Track的时候分配，
           并且只有在连续匹配上n_init帧才会转变为确定态。
        如果在处于不确定态的情况下没有匹配上任何detection，那将转变为删除态。
    Confirmed: 确定态，代表该Track确实处于匹配状态。如果当前Track属于确定态，但是失配连续达到max_age次数的时候，就会被转变为删除态。
    Deleted: 删除态，说明该Track已经失效。
    max_age: 代表一个Track存活期限，他需要和time_since_update变量进行比对。time_since_update是每次调用predict的时候就会+1，
    每次轨迹调用update函数的时候就会重置为0，,也就是说如果一个轨迹长时间没有update(没有匹配上)的时候，就会不断增加，
           直到time_since_update超过max_age(默认70)，将这个Track从Tracker中的列表删除。
    hits: 代表连续确认多少次，用在从不确定态转为确定态的时候。
          每次Track进行update的时候，hits就会+1, 如果hits>n_init(默认为3)，
          也就是连续三帧的该轨迹都得到了匹配，这时候才将不确定态转为确定态。
    需要说明的是每个轨迹还有一个重要的变量，features列表，存储该轨迹在不同帧对应位置通过ReID提取到的特征。
    为何要保存这个列表，而不是将其更新为当前最新的特征呢？这是为了解决目标被遮挡后再次出现的问题，需要从以往帧对应的特征进行匹配。
    另外，如果特征过多会严重拖慢计算速度，所以有一个参数budget用来控制特征列表的长度，取最新的budget个features,将旧的删除掉。
    """
class Track:
    # 一个轨迹的信息，包含(x,y,a,h) & v
    """
    A single target track with state space `(x, y, a, h)` and associated velocities,
    where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.
    纵横比
    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.

    """

    """
        mean:位置、速度状态分布均值向量，维度(8×1)
        convariance:位置、速度状态分布方差矩阵，维度(8×8)
        track_id:轨迹ID
        class_id:轨迹所属类别
        hits:轨迹更新次数(初始化为1)，即轨迹与目标连续匹配成功次数
        age:轨迹连续存在的帧数(初始化为1)，即轨迹出现到被删除的连续总帧数
        time_since_update:轨迹距离上次更新后的连续帧数(初始化为0)，即轨迹与目标连续匹配失败次数
        state:轨迹状态
        features:轨迹所属目标的外观语义特征，轨迹匹配成功时添加当前帧的新外观语义特征
        conf:轨迹所属目标的置信度得分
        _n_init:轨迹状态由不确定态到确定态所需连续匹配成功的次数
        _max_age:轨迹状态由不确定态到删除态所需连续匹配失败的次数
    """

    # max_age是一个存活期限，默认为70帧,在
    # Track(frame_id, mean, covariance, imm_controller, \
    #               self._next_id, self.n_init, self.max_age,
    #               detection.feature, detection_prob=0.9)
    def __init__(self,frame_id,mean, covariance, sita,imm_controller,track_id, n_init, max_age,
                 feature=None, detect_prob=0.9):
        self.cur_time_point_K = frame_id-1
        self.mean = mean
        self.covariance = covariance
        self.sita = sita
        self.mean_dict = {}
        self.covariance_dict = {}
        self.mean_dict[self.cur_time_point_K] = self.mean
        self.covariance_dict[self.cur_time_point_K] = self.covariance
        self.imm_controller = imm_controller
        # Pd
        self.detect_prob = detect_prob
        # 每个轨迹关联门内的检测索引列表，分为每个动态模型的列表，在cal_jpda_prob里初始化
        # self.associated_detection_list
        # self.distance_of_associated_detection_list
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.state = TrackState.Tentative
        self.features = []

        # 每个track对应多个features, 每次更新都将最新的feature添加到列表中
        if feature is not None:
            self.features.append(feature)


        self.math_operation = MathOperation()
        self._n_init = n_init
        # 如果连续n_init帧都没有出现匹配，设置为deleted状态
        self._max_age = max_age

        # _std_weight_position: 控制位置方差权重
        # _std_weight_velocity: 控制速度方差权重
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160


        """
        # hits和n_init进行比较
        # hits每次update的时候进行一次更新（只有match的时候才进行update）
        # hits代表匹配上了多少次，匹配次数超过n_init就会设置为confirmed状态
        self.age = 1 # 没有用到，和time_since_update功能重复
        self.time_since_update = 0
        # 每次调用predict函数的时候就会+1
        # 每次调用update函数的时候就会设置为0
        """

    # 6# def cal_volume_of_association_door(self,kalman_filter_P):

    # update_total_mean_and_covariance() 计算最后的X(k|k)和P(k|k)
    def update_total_mean_and_covariance(self):
        total_mean, total_covariance = self.imm_controller.cal_total_mean_and_covariance()
        self.mean_dict[self.cur_time_point_K+1] = total_mean
        self.covariance_dict[self.cur_time_point_K+1] = total_covariance
    def to_tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.
        Returns
        -------
        ndarray
            The bounding box.
        """
        # mean = [x,dx,y,dy,a,da,h,dh]
        # ret = [x,dx,y,dy]
        ret = self.mean[:4].copy()
        #w = ret[2]*ret[3]
        # y
        ret[1] = ret[2]
        # h
        ret[3] = self.mean[6]
        # w
        ret[2] = self.mean[4]*ret[3]
        ret[:2] -= ret[2:] / 2
        #x,y,w,h - -> ret[3] = h,, ret[2] = w/h,  ret[0] = x+w/2 , ret[1] = y +h/2
        #即可推断mean[:4]={目标框中心点坐标x, y, w/h, h}
        return ret
        # tlwh:目标框左上角横纵坐标x, y; 宽w; 高h

    def to_tlbr(self):
        #即得到左上角坐标，右下角坐标
        """Get current position in bounding box format `(min x, miny, max x,
        max y)`.
        Returns
        -------
        ndarray
            The bounding box.
        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    '''
    def predict(self, kf):
        # 预测下一帧轨迹信息
        # v.散播，宣传（观点、信仰等）；传播（运动、光线、声音等）；（动植物等）繁殖，使繁殖
        # n.分发；分销，配送；（电影在各院线的）发行，上映；分配，分布
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.
        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
    	# 预测下一帧轨迹信息
        self.mean, self.covariance = kf.predict(self.mean, self.covariance) #卡尔曼滤波预测下一帧轨迹的状态均值和方差
        self.increment_age() #调用函数，age+1，time_since_update+1
        """
        # age: 轨迹连续存在的帧数(初始化为1)，即轨迹出现到被删除的连续总帧数
        # time_since_update: 轨迹距离上次更新后的连续帧数(初始化为0)，即轨迹与目标连续匹配失败次数
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1
    '''


    def predict(self):
        # 预测下一帧轨迹信息
        # v.散播，宣传（观点、信仰等）；传播（运动、光线、声音等）；（动植物等）繁殖，使繁殖
        # n.分发；分销，配送；（电影在各院线的）发行，上映；分配，分布
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.
        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        '''
        预测下一帧轨迹信息
        '''
        self.mean, self.covariance = kf.predict(self.mean, self.covariance) #卡尔曼滤波预测下一帧轨迹的状态均值和方差
        self.increment_age() #调用函数，age+1，time_since_update+1
        """
        # age: 轨迹连续存在的帧数(初始化为1)，即轨迹出现到被删除的连续总帧数
        # time_since_update: 轨迹距离上次更新后的连续帧数(初始化为0)，即轨迹与目标连续匹配失败次数
        # 这步预测应该是得到x0j和p0j
        self.imm_controller.predict()
        self.imm_controller.kalman_filter_predict()
        self.age += 1
        self.time_since_update += 1

    def find_detections_in_association_Doors(self,detections,frame_idx,threshold=0.0):
        measure_index_list, measure_distance_list = self.imm_controller.find_detections_in_association_Doors(detections,self.sita,frame_idx)
        measure_index_list, measure_distance_list, model_ind= self.consolidate_same_measure_distance(measure_index_list, measure_distance_list)
        # 对多模型进行剪枝后的检测列表
        self.associated_detection_list = measure_index_list
        # 对多模型进行剪枝后的检测距离列表
        self.distance_of_associated_detection_list = measure_distance_list
        # 对多模型进行剪枝后的对应检测模型列表
        self.model_index_list = model_ind


    def consolidate_same_measure_distance(self, measure_index_list, measure_distance_list):
        # measure_index_list = [[model1_index],[model2_index],....]
        measure_index_temp = measure_index_list[0]
        measure_distance_temp = measure_distance_list[0]
        model_ind = np.zeros(len(measure_index_list[0]))
        for i in range(1,len(measure_index_list)):
            for j in range(len(measure_index_list[i])):
                if measure_index_list[i][j] not in measure_index_temp:
                    measure_index_temp.append(measure_index_list[i][j])
                    measure_distance_temp.append(measure_distance_list[i][j])
                    model_ind.append(i)
                else:
                    indx = measure_index_temp.index(measure_index_list[i][j])
                    model_ind[indx] = model_ind[indx] if measure_distance_temp[indx] <= measure_distance_list[i][j] \
                        else i
                    measure_distance_temp[indx] = measure_distance_temp[indx] if measure_distance_temp[indx] <= measure_distance_list[i][j] \
                        else measure_distance_list[i][j]
        return measure_index_temp,measure_distance_temp,model_ind


    def update_by_jpda(self,jpda_prob_matrix,detections,track_idx,detection_idx,frame_idx):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection : Detection
            The associated detection.
        """
        # find_detections_in_association_Doors
        cur_mean = self.mean.copy()
        cur_bxy = cur_mean[0:2]
        cur_bxy[1] = cur_mean[2]+(cur_mean[6]/2.)
        self.sita = self.math_operation.get_track_direction_sita(detections[detection_idx].to_bcxywh()[0:2],cur_bxy)
        self.mean, self.covariance = self.imm_controller.update_by_jpda(jpda_prob_matrix,detections,track_idx,frame_idx)
        # self.mean, self.covariance = kf.update(
        #     self.mean, self.covariance, detection.to_xyah())
        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def update_by_kalman(self,kf, detection):
        cur_mean = self.mean.copy()
        cur_bxy = cur_mean[0:2]
        cur_bxy[1] = cur_mean[2] + (cur_mean[6]/2.)
        self.sita = self.math_operation.get_track_direction_sita(detection.to_bcxywh()[0:2],cur_bxy)
        self.mean, self.covariance = kf.update(
            self.imm_controller.model_list[0].get_kalman_filter_X0()[0], self.imm_controller.model_list[0].get_kalman_filter_P0(),\
            detection.to_xyah())
        for model in self.imm_controller.model_list:
            model.set_update_mean_X(self.mean)
            model.set_update_covariance_P(self.covariance)

        self.features.append(detection.feature)
        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def update_on_state_vector(self,kf,track_idx,frame_idx):
        presumed_measurement = np.dot(self.imm_controller.model_list[0].get_kalman_filter_X0()[0],self.imm_controller.model_list[0].get_H_matrix().T)
        # R 测量过程中噪声的协方差
        std = [
            self._std_weight_position * presumed_measurement[3],
            self._std_weight_position * presumed_measurement[3],
            1e-1,
            self._std_weight_position * presumed_measurement[3]]
        presumed_measurement = presumed_measurement + np.square(std)
        self.mean, self.covariance = kf.update(
            self.imm_controller.model_list[0].get_kalman_filter_X0()[0],
            self.imm_controller.model_list[0].get_kalman_filter_P0(), presumed_measurement)
        for model in self.imm_controller.model_list:
            model.set_update_mean_X(self.mean)
            model.set_update_covariance_P(self.covariance)

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted
