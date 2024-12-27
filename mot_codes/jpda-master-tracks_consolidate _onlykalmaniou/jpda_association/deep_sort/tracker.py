# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
import copy
from sklearn.decomposition import PCA
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track
from .joint_prob_data_associate import Joint_prob_data_associate
from .imm_controller import Imm_Controller
from jpda_association.application_util.math_operation import MathOperation

class Tracker:
    """
    This is the multi-target tracker.
    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
    用于测量-跟踪关联的距离度量。
        A distance metric for measurement-to-track association.
    max_age : int
    最大错误匹配次数。
        Maximum number of missed misses before a track is deleted.
    n_init : int
    在确认航迹之前连续检测到的次数。如果在第一个“n init”帧内发生miss，轨道状态被设置为“Deleted”
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
    用于跟踪关联的测量的距离度量
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
    轨道保持在初始化阶段的帧数
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    # 是一个多目标tracker，保存了很多个track轨迹
    # 负责调用卡尔曼滤波来预测track的新状态+进行匹配工作+初始化第一帧
    # Tracker调用update或predict的时候，其中的每个track也会各自调用自己的update或predict

    # 调用的时候，后边的参数全部是默认的
    def __init__(self, metric, max_iou_distance=0.7, max_age=10, n_init=3,dynamic_model_list=['constant_velocity'],model_max_velocity=[8.0]):
        # metric是一个类，用于计算距离(余弦距离或马氏距离)
        # metric: nn_matching.NearestNeighborDistanceMetric
        # 用于测量 - 跟踪关联的距离度量。
        self.metric = metric
        # 最大iou，iou匹配的时候使用
        self.max_iou_distance = max_iou_distance
        # 直接指定级联匹配的cascade_depth参数
        self.max_age = max_age
        # n_init代表需要n_init次数的update才会将track状态设置为confirmed
        self.n_init = n_init

        # 卡尔曼滤波器
        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self.all_tracks = []
        # 下一个分配的轨迹id
        self._next_id = 1
        # 动态模型列表
        self.dynamic_model_list = dynamic_model_list
        self.math_operation = MathOperation()
        # self.unmatched_detetctionids = []
        self.unmatched_detetctions = []
        self.unmatched_detections_frame_id = -1
        self.new_tracks = []
        self.new_tracks_start_frame_id = -1
        # 模型最大速度
        self.model_max_velocity = model_max_velocity
        self.jpda_prob_calculator = Joint_prob_data_associate()
        # 记录每帧每个检测属于的轨迹
        self.detection_record = {}
        self.tracks_record = {}
        self.tracks_state = {}
        self.tracks_cant_deal = []
        self.deal_records = []

    """
    # metric是一个类，用于计算距离(余弦距离或马氏距离)
    self.max_iou_distance = max_iou_distance
    # 最大iou，iou匹配的时候使用
    self.max_age = max_age
    # 直接指定级联匹配的cascade_depth参数
    self.n_init = n_init
    # n_init代表需要n_init次数的update才会将track状态设置为confirmed
    self.kf = kalman_filter.KalmanFilter()# 卡尔曼滤波器
    self.tracks = [] # 保存一系列轨迹
    self._next_id = 1 # 下一个分配的轨迹id
    """

    def _initiate_track(self, new_tracks_start_frame_id,detectionstart_id, detection_start, detectionend_frameidx,detectionend_id,detection_end):
        dmean = detection_end.to_xyah()-detection_start.to_xyah()
        sita = self.math_operation.get_track_direction_sita(detection_end.to_bcxywh()[:2],detection_start.to_bcxywh()[:2])

        imm_controller = Imm_Controller(self.dynamic_model_list,self.model_max_velocity,\
                                                       deviation_process_noise = 0.5,deviation_measurement_noise=3,\
                                                       dynamic_model_transition_pro_matrix=[[1.0]])
        mean, covariance = imm_controller.initiate_dynamic_model(detection_end.to_xyah(),dmean)

        new_track = Track(new_tracks_start_frame_id, mean, covariance, sita,imm_controller, \
              self._next_id, self.n_init, self.max_age,detectionend_id,detection_end,detectionend_frameidx,
              detection_end.feature, detect_prob=0.9)
        self.detection_record['{}'.format(new_tracks_start_frame_id)]['{}'.format(detectionstart_id)].append(self._next_id)
        self.detection_record['{}'.format(detectionend_frameidx)]['{}'.format(detectionend_id)].append(
            self._next_id)
        self.tracks_record['{}'.format(self._next_id)] = {}
        self.tracks_state['{}'.format(self._next_id)] = False
        self.tracks_record['{}'.format(self._next_id)]['{}'.format(new_tracks_start_frame_id)] = {}
        self.tracks_record['{}'.format(self._next_id)]['{}'.format(detectionend_frameidx)] = {}

        self.tracks_record['{}'.format(self._next_id)]['{}'.format(new_tracks_start_frame_id)]\
            ['{}'.format(detectionstart_id)] = detection_start
        self.tracks_record['{}'.format(self._next_id)]['{}'.format(detectionend_frameidx)] \
            ['{}'.format(detectionend_id)] = detection_end
        self.tracks.append(new_track)
        self._next_id += 1

    # 遍历每个track都进行一次预测
    def start_new_tracks(self):
        # 为未匹配的检测初始化轨迹
        for detectionstart_frameid,detectionstart_id,detection_start,detectionend_frameidx,detectionend_id,detection_end in self.new_tracks:
            self._initiate_track(detectionstart_frameid,detectionstart_id, detection_start,\
                                 detectionend_frameidx,detectionend_id, detection_end)
        self.new_tracks = []
        self.new_tracks_start_frame_id = -1

    def pdist(self,a, b, threshold):
        # 译文：计算“a”和“b”点之间的成对距离的平方
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

    def aera_dist(self,a, b, threshold):
        # 拷贝一份数据
        a, b = np.asarray(a), np.asarray(b)
        if len(a) == 0 or len(b) == 0:
            # (N,0) or (0,L)
            return np.zeros((len(a), len(b)))
        # a: NM; b: LM
        # a2:N ; b2:L
        awh, bwh = a[:,0]*a[:,1],b[:,0]*b[:,1]
        r2 = awh[:, None] - bwh[None, :]
        r2 = np.abs(r2)
        r2 = np.sqrt(r2)
        # np.inf :+oo
        # r2 = np.clip(r2, 0., threshold)
        return r2


    def _nn_cosine_distance(self, x, y, threshold_distance=6,threshold_aera=8):
        # 计算余弦距离
        y_cxy = [det.to_xyah()[:2] for det in y]
        y_wh = [det.tlwh[2:4] for det in y]
        distances = self.pdist([x.to_xyah()[0:2]], y_cxy, threshold_distance)
        area = self.aera_dist([x.tlwh[2:4]], y_wh, threshold_aera)
        total_distance = distances + area
        # detection_index = []
        # distance_detection_and_target = []
        # for detection_i in range(distances.shape[1]):
        #     if distances[0, detection_i] < threshold:
        #         detection_index.append(detection_i)
        #         distance_detection_and_target.append(distances[0, detection_i])
        # return detection_index, distance_detection_and_target
        return total_distance

    def get_cost_matrix(self,detections,next_frame_detections):
        cost_matrix = np.zeros((len(detections),len(next_frame_detections)))
        for i, detection in enumerate(detections):
            cost_matrix[i, :] = self._nn_cosine_distance(detections[i], next_frame_detections)
        return cost_matrix

    def start_tracks(self,detections,frame_idx,threshold=500):
        # 为第一帧初始化轨迹
        if len(detections)==0:
            print("the tracking process is end!!!")
            return
        if len(self.unmatched_detetctions) == 0:
            return  [i for i in range(len(detections))]
        unmatched_detections_frame_id = self.unmatched_detections_frame_id
        # self.unmatched_detetctions.append([a_detection_idx, detections[a_detection_idx]])
        last_frame_detections = [detection for a_detection_idx, detection in self.unmatched_detetctions]
        last_frame_detection_ids = [a_detection_idx for a_detection_idx, detection in self.unmatched_detetctions]
        dealed_detections = []
        # self.new_tracks_start_frame_id = unmatched_detections_frame_id
        cost_matrix = self.get_cost_matrix(last_frame_detections,detections)
        for last_frame_detection_index in range(cost_matrix.shape[0]):
            # self.detection_record['{}'.format(unmatched_detections_frame_id)]\
            #     ['{}'.format(last_frame_detection_ids[last_frame_detection_index])] = []
            for detectionid in range(cost_matrix.shape[1]):
                if cost_matrix[last_frame_detection_index,detectionid] <threshold:
                    self.new_tracks.append([unmatched_detections_frame_id,last_frame_detection_ids[last_frame_detection_index],\
                                            last_frame_detections[last_frame_detection_index],\
                                            frame_idx,detectionid,detections[detectionid]])
                    if detectionid not in dealed_detections:
                        dealed_detections.append(detectionid)
                    # matched_detections.append(detectionid)
        undealed_detections = []
        for a_detectionid in range(len(detections)):
            if a_detectionid not in dealed_detections:
                undealed_detections.append(a_detectionid)
        self.unmatched_detetctions = []
        self.unmatched_detections_frame_id = -1
        return undealed_detections
        # self.unmatched_detetctions_frame_id = frameid + 1
        # for a_next_frame_detectionid in range(len(next_frame_detections)):
        #     if a_next_frame_detectionid not in matched_detections:
        #         # 对于未分配的检测，从这里开始初始化轨迹
        #         self.unmatched_detetctions.append([a_next_frame_detectionid, next_frame_detections[a_next_frame_detectionid]])


    def initiate_a_new_track(self, detectionid,detection,next_frame_detectionid,next_frame_detection,frame_id):
        dmean = next_frame_detection.to_xyah()-detection.to_xyah()
        sita = self.math_operation.get_track_direction_sita(next_frame_detection.to_bcxywh()[:2],detection.to_bcxywh()[:2])
        imm_controller = Imm_Controller(self.dynamic_model_list,self.model_max_velocity,\
                                                       deviation_process_noise = 0.5,deviation_measurement_noise=7,\
                                                       dynamic_model_transition_pro_matrix=[[1.0]])
        # 这里初始化时传入的是下一帧检测，因为已经实现了下一帧检测与当前帧检测的分配，因此下一帧的检测才未当前轨迹的最新状态位置
        mean, covariance = imm_controller.initiate_dynamic_model(next_frame_detection.to_xyah(),dmean)
        new_track = Track(frame_id, mean, covariance, sita,imm_controller, \
              self._next_id, self.n_init, self.max_age,curdetection_id=next_frame_detectionid,curdetection_mean=next_frame_detection,
              feature=next_frame_detection.feature,detect_prob=0.9)
        self.tracks.append(new_track)
        self._next_id += 1
        return self._next_id-1

    # 遍历每个track都进行一次预测
    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict()
            # track.predict(self.kf)

    def cal_jpda_prob(self,detections,frame_idx,threshold=0.0):
        for track in self.tracks:
            track.find_detections_in_association_Doors_by_iou(detections, frame_idx, threshold)
            # track.find_detections_in_association_Doors(detections,frame_idx,threshold)
        # if len(self.tracks)>0:
        #     self.jpda_prob_calculator.update_tracks_and_detections(self.tracks,detections,threshold)
        #     # 根据每个track的关联域生成track与detections的关联矩阵
        #     self.jpda_prob_calculator.generate_associate_matrix(len(self.dynamic_model_list))
        #     # 生成联合事件（连通图）
        #     associate_events = self.jpda_prob_calculator.generate_associate_events()
        #     # 联合概率矩阵
        #     jpda_prob_matrix = np.zeros((len(detections)+1,len(self.tracks)),dtype=np.float64)
        #
        #     for a_associate_event in associate_events:
        #         if len(a_associate_event['tracks'])!=0 and len(a_associate_event['detections'])!=0:
        #             # if len(a_associate_event['tracks']) ==1 and len(a_associate_event['detections'])==1:
        #             #     continue
        #             # else:
        #             associate_events_jpda_prob_matrix = np.zeros((len(detections) + 1, len(self.tracks)), dtype=np.float64)
        #             # 对这个联合事件生成可行事件列表
        #             associate_events_detections,associate_events_tracks,associate_event_matrix_list = \
        #                 self.jpda_prob_calculator.generate_associate_event_matrixs(a_associate_event['tracks'], a_associate_event['detections'])
        #             #为当前联合事件中的可行事件计算beta(jt)
        #             jpda_prob_matrix += self.jpda_prob_calculator.cal_beta_of_cur_associate_event(associate_events_jpda_prob_matrix,\
        #                                              associate_events_detections,associate_events_tracks,associate_event_matrix_list)
        #     return jpda_prob_matrix
        # else:
        #     return None
    # def consolidate_detection_records(self,frame_id):
    #     if frame_id >1:
    #         cur_frame_det_records = self.detection_record["{}".format(frame_id)]
    #         temp_record = copy.deepcopy(cur_frame_det_records)
    #         last_frame_det_records = self.detection_record["{}".format(frame_id-1)]
    #         for detection_id in cur_frame_det_records.keys():
    #             if len(cur_frame_det_records[detection_id]) >1:
    #                 for last_detection_id in last_frame_det_records.keys():
    #                     if last_frame_det_records[last_detection_id] == cur_frame_det_records[detection_id]:
    #                         temp_record.pop(detection_id)
    #         self.detection_record["{}".format(frame_id)] = temp_record

    def match(self, detections, undealed_detections, next_frame_detections, frame_idx):
        # Run matching cascade.
        # 通过jpda矩阵匹配一部分
        matches_a, unmatched_tracks, unmatched_detections = self.match_by_iou(detections)
        # 应该首先进行关联事件计算,每个轨迹进行计算，每个轨迹应该存储在关联门内的每个量测
        # for track_idx,track in enumerate(self.tracks):
        #     track.update(jpda_prob_matrix,detections,track_idx,frame_idx)
        # 1. 针对匹配上的结果
        # Update track set.
        # for track_idx, detection_idx in matches:
        #     self.tracks[track_idx].update(
        #         self.kf, detections[detection_idx])
        # 对所有可能的匹配初始化轨迹
        # zero_detection_tracks 轨迹没有检测
        zero_detection_tracks = []
        deald_detections = []
        # if len(self.tracks)==0:
        #     for a_detection_idx in unmatched_detections:
        #         self.unmatched_detetctions.append([a_detection_idx,detections[a_detection_idx]])
        #     self.unmatched_detections_frame_id = frame_idx
        #     return
        # else:
        for idx, track in enumerate(self.tracks):
            # # 对多模型进行剪枝后的检测列表
            # self.associated_detection_list = measure_index_list
            # # 对多模型进行剪枝后的检测距离列表
            # self.distance_of_associated_detection_list = measure_distance_list
            # # 对多模型进行剪枝后的对应检测模型列表
            # self.model_index_list = model_ind
            if len(track.associated_detection_list) == 0:
                # 关联门内没有检测
                zero_detection_tracks.append(idx)
            else:
                # 关联门内有检测
                # 因为上面已经通过jpda算法进行了一次匹配，这里将未匹配的检测初始为新的轨迹，需要去除已经匹配的检测
                deald_detections += track.associated_detection_list
                a_pair_of_match = [-1, -1]
                for track_idx, detection_idx in matches_a:
                    if track_idx == idx:
                        # self.detection_record['{}'.format(track.curdetection_frameid)] \
                        #     ['{}'.format(track.curdetection_id)].append(track.track_id)
                        self.detection_record['{}'.format(frame_idx)] \
                            ['{}'.format(detection_idx)].append(track.track_id)
                        self.tracks_record['{}'.format(track.track_id)]['{}'.format(frame_idx)] = {}
                        self.tracks_record['{}'.format(track.track_id)]['{}'.format(frame_idx)] \
                            ['{}'.format(detection_idx)] = detections[detection_idx]
                        a_pair_of_match = [track_idx, detection_idx]
                        break
                removed_matched_detection = [associated_detetction for associated_detetction in \
                                             track.associated_detection_list if
                                             associated_detetction != a_pair_of_match[1]]
                # 开始为未匹配的检测初始化轨迹
                # self.detection_record['{}'.format(track.curdetection_frameid)]['{}'.format(track.curdetection_id)] = []
                # self.new_tracks_start_frame_id = track.curdetection_frameid
                for unmatched_detectionid in removed_matched_detection:
                    self.new_tracks.append(
                        [track.curdetection_frameid, track.curdetection_id, track.curdetection_mean, frame_idx,
                         unmatched_detectionid, \
                         detections[unmatched_detectionid]])
                    # track_idx = self.initiate_a_new_track(track.curdetection_id, detections[detectionid],track.curdetection_mean
                    #                                                   unmatched_detectionid, \
                    #                                                   next_frame_detections[unmatched_detectionid],
                    #                                                   frame_idx)
                    # self.detection_record['{}'.format(frame_idx)]['{}'.format(track.curdetection_id)].append(track_idx)
        # 对于已经匹配的轨迹和检测，进行状态更新
        # for track_idx, detection_idx in matches_a:
        #     self.tracks[track_idx].update_by_jpda(jpda_prob_matrix, detections, track_idx, detection_idx, frame_idx)
        for track_idx, detection_idx in matches_a:
            # self.detection_record['{}'.format(track.curdetection_frameid)]['{}'.format(track.curdetection_id)] = []
            self.tracks[track_idx].update_by_kalman(self.kf, detections[detection_idx],detection_idx,frame_idx)
        # 对于关联门内没有检测的轨迹，归类为未匹配的轨迹
        for track_idx in zero_detection_tracks:
            # 根据关联结果改变轨迹状态
            # self.tracks[track_idx].update_on_state_vector(self.kf, track_idx, frame_idx)
            # self.tracks[track_idx].update_by_presumed_measurement(track_idx, frame_idx)
            self.tracks[track_idx].mark_missed()
        # 将用于初始化轨迹后剩余的检测与未关联的检测合并
        consolidated_undealed_detections = []
        for a_detetcionid in undealed_detections:
            if a_detetcionid not in deald_detections:
                consolidated_undealed_detections.append(a_detetcionid)
        self.unmatched_detections_frame_id = frame_idx
        # 对于未分配的检测，从这里开始初始化轨迹
        for a_detection_idx in consolidated_undealed_detections:
            self.unmatched_detetctions.append([a_detection_idx, detections[a_detection_idx]])

        # for detection_idx in unmatched_detections:
        #     self.unmatched_detetctions.append(detections[detection_idx])
        # for detection_idx in unmatched_detections:
        #     self._initiate_track(detections[detection_idx],next_frame_detections,frame_idx)
        track_not_end = []
        for tra in self.tracks:
            if tra.is_deleted():
                self.tracks_state['{}'.format(tra.track_id)] = True
            else:
                track_not_end.append(tra)
        self.tracks = track_not_end
        # self.tracks = [t for t in self.tracks if not t.is_deleted()]
        # Update distance metric.
        # active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        # features, targets = [], []
        # for track in self.tracks:
        #     if not track.is_confirmed():
        #         continue
        #     # 只处理Tentative = 1不确定态的轨迹
        #     # confirmed了的轨迹，
        #     # 每个轨迹对应多个feature
        #     features += track.features
        #     # 有几组特征，初始化一个轨迹Id
        #     targets += [track.track_id for _ in track.features]
        #     track.features = []
        # # 更新nn_match中的sample字典的激活的轨迹
        # self.metric.partial_fit(
        #     np.asarray(features), np.asarray(targets), active_targets)

    # 进行测量的更新和轨迹管理
    def update(self,jpda_prob_matrix,detections,undealed_detections,next_frame_detections,frame_idx):
        """Perform measurement update and track management.
        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.
        """
        # Run matching cascade.
        # 通过jpda矩阵匹配一部分
        self.detection_record['{}'.format(frame_idx)] = {}
        matches_a,unmatched_tracks, unmatched_detections = self._match(jpda_prob_matrix,detections)
        # 应该首先进行关联事件计算,每个轨迹进行计算，每个轨迹应该存储在关联门内的每个量测
        # for track_idx,track in enumerate(self.tracks):
        #     track.update(jpda_prob_matrix,detections,track_idx,frame_idx)
        # 1. 针对匹配上的结果
        # Update track set.
        # for track_idx, detection_idx in matches:
        #     self.tracks[track_idx].update(
        #         self.kf, detections[detection_idx])
        # 对所有可能的匹配初始化轨迹
        # zero_detection_tracks 轨迹没有检测
        zero_detection_tracks = []
        deald_detections = []
        # if len(self.tracks)==0:
        #     for a_detection_idx in unmatched_detections:
        #         self.unmatched_detetctions.append([a_detection_idx,detections[a_detection_idx]])
        #     self.unmatched_detections_frame_id = frame_idx
        #     return
        # else:
        for idx,track in enumerate(self.tracks):
                # # 对多模型进行剪枝后的检测列表
                # self.associated_detection_list = measure_index_list
                # # 对多模型进行剪枝后的检测距离列表
                # self.distance_of_associated_detection_list = measure_distance_list
                # # 对多模型进行剪枝后的对应检测模型列表
                # self.model_index_list = model_ind
                if len(track.associated_detection_list) == 0:
                    # 关联门内没有检测
                    zero_detection_tracks.append(idx)
                else:
                    # 关联门内有检测
                    # 因为上面已经通过jpda算法进行了一次匹配，这里将未匹配的检测初始为新的轨迹，需要去除已经匹配的检测
                    deald_detections += track.associated_detection_list
                    a_pair_of_match = [-1,-1]
                    for track_idx, detection_idx in matches_a:
                        if track_idx == idx:
                            a_pair_of_match = [track_idx,detection_idx]
                            break
                    removed_matched_detection = [associated_detetction for associated_detetction in \
                                                 track.associated_detection_list if associated_detetction != a_pair_of_match[1]]
                    # 开始为未匹配的检测初始化轨迹
                    self.detection_record['{}'.format(track.curdetection_frameid)]['{}'.format(track.curdetection_id)] = []
                    # self.new_tracks_start_frame_id = track.curdetection_frameid
                    for unmatched_detectionid in removed_matched_detection:
                        self.new_tracks.append([track.curdetection_frameid,track.curdetection_id,track.curdetection_mean,frame_idx,unmatched_detectionid,\
                                                detections[unmatched_detectionid]])
                        # track_idx = self.initiate_a_new_track(track.curdetection_id, detections[detectionid],track.curdetection_mean
                        #                                                   unmatched_detectionid, \
                        #                                                   next_frame_detections[unmatched_detectionid],
                        #                                                   frame_idx)
                        # self.detection_record['{}'.format(frame_idx)]['{}'.format(track.curdetection_id)].append(track_idx)
        #对于已经匹配的轨迹和检测，进行状态更新
        for track_idx, detection_idx in matches_a:
            self.tracks[track_idx].update_by_jpda(jpda_prob_matrix,detections,track_idx,detection_idx,frame_idx)
        # for track_idx, detection_idx in matches_b:
        #     self.tracks[track_idx].update_by_kalman(self.kf, detections[detection_idx])
        # 对于关联门内没有检测的轨迹，归类为未匹配的轨迹
        for track_idx in zero_detection_tracks:
            # 根据关联结果改变轨迹状态
            self.tracks[track_idx].update_on_state_vector(self.kf,track_idx,frame_idx)
            # self.tracks[track_idx].update_by_presumed_measurement(track_idx, frame_idx)
            self.tracks[track_idx].mark_missed()
        # 将用于初始化轨迹后剩余的检测与未关联的检测合并
        consolidated_undealed_detections = []
        for a_detetcionid in undealed_detections:
            if a_detetcionid not in deald_detections:
                consolidated_undealed_detections.append(a_detetcionid)
        self.unmatched_detections_frame_id = frame_idx
        #对于未分配的检测，从这里开始初始化轨迹
        for a_detection_idx in consolidated_undealed_detections:
                self.unmatched_detetctions.append([a_detection_idx,detections[a_detection_idx]])

        # for detection_idx in unmatched_detections:
        #     self.unmatched_detetctions.append(detections[detection_idx])
        # for detection_idx in unmatched_detections:
        #     self._initiate_track(detections[detection_idx],next_frame_detections,frame_idx)

        self.tracks = [t for t in self.tracks if not t.is_deleted()]
        # Update distance metric.
        # active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        # features, targets = [], []
        # for track in self.tracks:
        #     if not track.is_confirmed():
        #         continue
        #     # 只处理Tentative = 1不确定态的轨迹
        #     # confirmed了的轨迹，
        #     # 每个轨迹对应多个feature
        #     features += track.features
        #     # 有几组特征，初始化一个轨迹Id
        #     targets += [track.track_id for _ in track.features]
        #     track.features = []
        # # 更新nn_match中的sample字典的激活的轨迹
        # self.metric.partial_fit(
        #     np.asarray(features), np.asarray(targets), active_targets)

    def match_by_iou(self, detections):
        matches_track_det = []
        unmatched_tracks = []
        unmatched_detections = []
        matches_detection = []
        for trackid,track in enumerate(self.tracks):
            associated_detections_list = track.associated_detection_list
            associated_detections_distance = track.distance_of_associated_detection_list
            if len(associated_detections_list)==0:
                unmatched_tracks.append(trackid)
            else:
                min_distance = max(associated_detections_distance)
                index = np.argmax(associated_detections_distance)
                matches_track_det.append([trackid,associated_detections_list[index]])
                matches_detection.append(associated_detections_list[index])
        for detection_idx in range(len(detections)):
            if detection_idx not in matches_detection:
                unmatched_detections.append(detection_idx)
        return matches_track_det,unmatched_tracks, unmatched_detections

    # 主要功能是进行匹配，找到匹配的，未匹配的部分
    def _match(self, jpda_prob_matrix,detections):
        # 功能： 用于计算track和detection之间的距离，代价函数
        #        需要使用在KM算法之前
        # 调用：
        # cost_matrix = distance_metric(tracks, detections,
        #                  track_indices, detection_indices)
        # 在linear_assignment的min_cost_matching里回调
        # def gated_metric(tracks, dets, track_indices, detection_indices):
        #     features = np.array([dets[i].feature for i in detection_indices])
        #     # 轨迹ID列表
        #     targets = np.array([tracks[i].track_id for i in track_indices])
        #     # 1. 通过最近邻计算出代价矩阵 cosine distance
        #     cost_matrix = self.metric.distance(features, targets)
        #     # 2. 计算马氏距离,得到新的状态矩阵
        #     cost_matrix = linear_assignment.gate_cost_matrix(
        #         self.kf, cost_matrix, tracks, dets, track_indices,
        #         detection_indices)
        #
        #     return cost_matrix

        # # Split track set into confirmed and unconfirmed tracks.
        # confirmed_tracks = [
        #     i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        # unconfirmed_tracks = [
        #     i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        # matches_a, unmatched_tracks_a, unmatched_detections = \
        #     linear_assignment.matching_cascade(gated_metric, self.metric.matching_threshold,
        #                                        self.max_age, self.tracks, detections, jpda_prob_matrix, confirmed_tracks)
        track_inds = [i for i, t in enumerate(self.tracks)]
        matches_a, unmatched_tracks_a, unmatched_detections_a = \
             linear_assignment.matching_by_jpda_matrix(self.metric.matching_threshold, \
                                                       self.tracks, detections, jpda_prob_matrix, track_inds)
        # Associate remaining tracks together with unconfirmed tracks
        # using IOU.
        # iou_track_candidates = [
        #     k for k in unmatched_tracks_a]
        # matches_b, unmatched_tracks_b, unmatched_detections_b = \
        #     linear_assignment.min_cost_matching(
        #         iou_matching.iou_cost, self.max_iou_distance, self.tracks,
        #         detections, iou_track_candidates)
        # matches = matches_a + matches_b
        # unmatched_detections = []
        # matches_detections_b = []
        # for track_idx, detection_idx in matches_b:
        #     matches_detections_b.append(detection_idx)
        # for detection_idx in unmatched_detections_a:
        #     if detection_idx not in matches_detections_b:
        #         unmatched_detections.append(detection_idx)
        # unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches_a,unmatched_tracks_a, unmatched_detections_a

    def get_dv_by_iou_distance(self,cur_detection,next_frame_detections):
        if len(next_frame_detections)==0:
            return cur_detection
        cost_matrix = np.zeros((1, len(next_frame_detections)))
        bbox = cur_detection.tlwh
        candidates = np.asarray([det.tlwh for det in next_frame_detections])
        cost_matrix[0,:] = 1. - iou_matching.iou(bbox, candidates)
        max_ind = 0
        for i in range(len(next_frame_detections)):
            if cost_matrix[0,i] == min(cost_matrix[0, :]):
                max_ind = i
                # print(candidates[i])
                break
        return next_frame_detections[max_ind]

    def remove_fake_tracks(self,track_len):
        for track_id in self.tracks_record.keys():
            if len(self.tracks_record[track_id].keys()) < track_len and  self.tracks_state[track_id] == True:
                self.tracks_state.pop(track_id)

    def get_unnormal_points(self):
        # 去除明显错误轨迹，长度小于3且状态为已经结束
        temp_dict = copy.deepcopy(self.tracks_record)
        tra_temp = []
        for track in self.tracks_cant_deal:
            for t in track:
                if t not in tra_temp:
                    tra_temp.append(t)
        for track_id in self.tracks_record.keys():
            if len(self.tracks_record[track_id].keys()) < 3 and self.tracks_state[track_id] == True and int(track_id) not in tra_temp:
                temp_dict.pop(track_id)
                self.tracks_state.pop(track_id)

        self.tracks_record = copy.deepcopy(temp_dict)
        # 对检测记录进行更新
        temp_dict = copy.deepcopy(self.detection_record)
        for frame_id in self.detection_record.keys():
            for detection_id in self.detection_record[frame_id].keys():
                track_ids = self.detection_record[frame_id][detection_id]
                for track_id in track_ids:
                    if "{}".format(track_id) not in self.tracks_record.keys():
                        ind = temp_dict[frame_id][detection_id].index(track_id)
                        temp_dict[frame_id][detection_id].pop(ind)
                        if track_ids in self.tracks_cant_deal:
                            self.tracks_cant_deal.remove(track_ids)
        self.detection_record = copy.deepcopy(temp_dict)
        return temp_dict
    def cal_track_feature(self,head_tracks):
        # head_tracks为一段前段轨迹，由按帧顺序排列的检测组成列表
        posis_xy =np.array([det.to_cxywh()[:2] for det in head_tracks])
        pca = PCA(n_components=1)  # 降到 1 维
        pca.fit(posis_xy)
        A = pca.transform(posis_xy)  # 降维后的结果
        # 得到特征值
        eigenvalues_A = pca.explained_variance_
        # 得到特征数量
        eigenvectors_A = pca.components_
        # 由于此特征向量不区分轨迹的前进方向，需要进一步求轨迹的方向
        first_posi = posis_xy[0]
        final_posi = posis_xy[-1]
        direction = [0,0]
        # x坐标的变化方向
        if first_posi[0] < final_posi[0]:
            # 正方向变化
            direction[0] = 1
        elif first_posi[0] == final_posi[0]:
            direction[0] = 0
        else:
            direction[0] = -1
        # y坐标的变化方向
        if first_posi[1] < final_posi[1]:
            # 正方向变化
            direction[1] = 1
        elif first_posi[1] == final_posi[1]:
            direction[1] = 0
        else:
            direction[1] = -1
        # 根据轨迹方向调整特征向量
        # x方向
        if direction[0] > 0:
            if eigenvectors_A[0,0] < 0:
                eigenvectors_A[0,0] = -1*eigenvectors_A[0,0]
        elif direction[0] < 0:
            if eigenvectors_A[0,0] > 0:
                eigenvectors_A[0,0] = -1*eigenvectors_A[0,0]
        # y方向
        if direction[1] > 0:
            if eigenvectors_A[0, 1] < 0:
                eigenvectors_A[0, 1] = -1 * eigenvectors_A[0, 1]
        elif direction[1] < 0:
            if eigenvectors_A[0, 1] > 0:
                eigenvectors_A[0, 1] = -1 * eigenvectors_A[0, 1]
        return eigenvectors_A
    def match_bet_tracks(self,head_tracks,tail_track_list):
        # head_tracks为一段前段轨迹，由按帧顺序排列的检测组成列表
        # tail_track_list为后段轨迹列表，由多段轨迹组成
        head_pca_vector = self.cal_track_feature(head_tracks[0:])
        macth_angles = [180. for _ in range(len(tail_track_list))]
        for a_track_ind in range(len(tail_track_list)):
            if len(tail_track_list[a_track_ind]) <2:
                continue
            atrack_pca_vector = self.cal_track_feature(tail_track_list[a_track_ind][0:])
            match_rate = self.cal_match_rate(head_pca_vector,atrack_pca_vector)
            macth_angles[a_track_ind] = match_rate[0,0]
        return macth_angles
    def cal_match_rate(self,head_pca_vector, behind_pca_vector):
        #方向上合理，可以继续
        sita = np.arccos(np.dot(head_pca_vector, behind_pca_vector.T) / (
                        np.sqrt(head_pca_vector[0, 0] ** 2 + head_pca_vector[0, 1] ** 2) \
                        * np.sqrt(behind_pca_vector[0, 0] ** 2 + behind_pca_vector[0, 1] ** 2)))*180/np.pi
        return sita

    def post_process_per_n_frames(self,temp_detection_dict,frame_id, interval, track_episode_length):
    # def post_process_per_n_frames(self,frame_id,interval,track_episode_length):
        # 遍历frame索引
        # frame_id_keys = ['{}'.format() for id in range(frame_id-interval+1,frame_id)]
        # 对于发生了轨迹号更换的轨迹，需要相应地变化self.detection_record中记录的轨迹号
        # 更新处理过的轨迹
        # 如果相邻两帧对轨迹进行了两次交换
        # 比如，2（25，9）（25，9）
        # 而轨迹25断开，形成（34，25）
        # 下一帧变为（34，9）
        # 34轨迹再次断开，（35，34）则34只是一个过渡，应该变为最终（35，25）
        def replace(temp_temp_detection_record, temp_temp_detection_dict,tempA,frameid):
            for cur,tempAf in tempA:
                # 处理temp_temp_detection_dict
                detection_dict_sorted_index = sorted(temp_temp_detection_dict.keys(), key=lambda k: int(k))
                cur_frame_ind = detection_dict_sorted_index.index(frame_id)
                for update_frame_ind in detection_dict_sorted_index[cur_frame_ind + 1:]:
                    for a_detection in temp_temp_detection_dict[update_frame_ind].keys():
                        if tempAf in temp_temp_detection_dict[update_frame_ind][a_detection]:
                            flag = 0
                            if cur in temp_temp_detection_dict[update_frame_ind][a_detection]:
                                flag = 1
                            ind = temp_temp_detection_dict[update_frame_ind][a_detection].index(tempAf)
                            temp_temp_detection_dict[update_frame_ind][a_detection][ind] = cur
                            if flag == 1:
                                lab = 0
                                for tr in temp_temp_detection_dict[update_frame_ind][a_detection]:
                                    if tr == cur:
                                        lab += 1
                                if len(temp_temp_detection_dict[update_frame_ind][a_detection]) == lab:
                                    temp_temp_detection_dict[update_frame_ind][a_detection] = [cur]
                # 处理temp_temp_detection_record
                detection_dict_sorted_index = sorted(temp_temp_detection_record.keys(), key=lambda k: int(k))
                cur_frame_ind = detection_dict_sorted_index.index(frame_id)
                for update_frame_ind in detection_dict_sorted_index[cur_frame_ind + 1:]:
                    for a_detection in temp_temp_detection_record[update_frame_ind].keys():
                        if tempAf in temp_temp_detection_record[update_frame_ind][a_detection]:
                            flag = 0
                            if cur in temp_temp_detection_record[update_frame_ind][a_detection]:
                                flag = 1
                            ind = temp_temp_detection_record[update_frame_ind][a_detection].index(tempAf)
                            temp_temp_detection_record[update_frame_ind][a_detection][ind] = cur
                            if flag == 1:
                                lab = 0
                                for tr in temp_temp_detection_record[update_frame_ind][a_detection]:
                                    if tr == cur:
                                        lab += 1
                                if len(temp_temp_detection_record[update_frame_ind][a_detection]) == lab:
                                    temp_temp_detection_record[update_frame_ind][a_detection] = [cur]
                # 处理self.tracks_cant_deal
                for track_pair_ind in range(len(self.tracks_cant_deal)):
                    for track_ind in range(len(self.tracks_cant_deal[track_pair_ind])):
                        if self.tracks_cant_deal[track_pair_ind][track_ind] == tempAf:
                            self.tracks_cant_deal[track_pair_ind][track_ind] = cur
            return temp_temp_detection_dict,temp_temp_detection_record

        def update_track_cant_deal(cur_track,origin_track):
            for track_pair_ind in range(len(self.tracks_cant_deal)):
                for track_ind in range(len(self.tracks_cant_deal[track_pair_ind])):
                    if self.tracks_cant_deal[track_pair_ind][track_ind] == origin_track:
                        self.tracks_cant_deal[track_pair_ind][track_ind] = cur_track


        def update_tracks(dealed_track_reord, cur_track,origin_track):
            temp_track_pair = []
            lab = 0
            for head,tail in dealed_track_reord:
                if head == origin_track:
                    temp_track_pair.append([cur_track,tail])
                    lab = 1
                else:
                    temp_track_pair.append([head,tail])
            if lab == 0:
                temp_track_pair.append([cur_track,origin_track])
            return temp_track_pair

        def update_detection_record(detection_dict, origin_track, cur_track,frame_id):
            detection_dict_sorted_index = sorted(detection_dict.keys(), key=lambda k: int(k))
            cur_frame_ind = detection_dict_sorted_index.index(frame_id)
            for update_frame_ind in detection_dict_sorted_index[cur_frame_ind+1:]:
                for a_detection in detection_dict[update_frame_ind].keys():
                    if origin_track in detection_dict[update_frame_ind][a_detection]:
                        flag = 0
                        if cur_track in detection_dict[update_frame_ind][a_detection]:
                            flag = 1
                        ind = detection_dict[update_frame_ind][a_detection].index(origin_track)
                        detection_dict[update_frame_ind][a_detection][ind] = cur_track
                        if flag == 1:
                            lab = 0
                            for tr in detection_dict[update_frame_ind][a_detection]:
                                if tr == cur_track:
                                    lab +=1
                            if len(detection_dict[update_frame_ind][a_detection]) == lab:
                                detection_dict[update_frame_ind][a_detection] = [cur_track]
            return detection_dict

        dealed_track_reord = []
        for frame_id in temp_detection_dict.keys():
            dealed_track_reord_frame = []
            # 遍历检测索引
            lab = 0
            for detection_id in temp_detection_dict[frame_id].keys():
                if len(temp_detection_dict[frame_id][detection_id])>1:
                    # 这里复制一份是为了在轨迹交换时使用
                    #大于1，说明发生了轨迹分裂,当前检测属于多个轨迹
                    head_track_list = []
                    tail_track_list = []
                    track_is_end = []
                    # 轨迹片段的长度
                    track_episode_length = track_episode_length
                    # 取轨迹
                    tracks = temp_detection_dict[frame_id][detection_id]
                    # 对每个轨迹进行处理
                    # 第47帧出现101
                    if tracks[0]==101 and tracks[1] ==4 :
                        print("s")
                    # if tracks == [101,101]:
                    #     print("s")
                    for track_id in temp_detection_dict[frame_id][detection_id]:
                        # track_item是一条轨迹，包含以帧号为键，检测号为值的字典
                        track_item = self.tracks_record['{}'.format(track_id)]
                        # 需要先对轨迹中的检测按帧顺序进行排序
                        track_sorted_index = sorted(track_item.keys(), key=lambda k: int(k))
                        # 取出当前检测在轨迹中的索引位置
                        if  "{}".format(49) == frame_id :
                            print("s")
                        # print(frame_id)
                        # print(track_id)
                        detection_posi_index = track_sorted_index.index(frame_id)
                        head_index_list = None
                        tail_index_list = None
                        # 从当前检测在轨迹中的索引位置，取轨迹长
                        if detection_posi_index+1 >= track_episode_length:
                             head_index_list = track_sorted_index[(detection_posi_index+1)-track_episode_length:detection_posi_index+1]
                        else:
                            head_index_list = track_sorted_index[0:detection_posi_index+1]
                        if detection_posi_index+1+track_episode_length <= len(track_sorted_index):
                            tail_index_list = track_sorted_index[detection_posi_index:detection_posi_index+track_episode_length]
                        else:
                            tail_index_list = track_sorted_index[detection_posi_index:]
                        track_head_list = []
                        track_tail_list = []
                        # 根据检测索引取出检测
                        for head_index in head_index_list:
                            frame_detection = track_item[head_index]
                            detection_idxs = list(frame_detection.keys())
                            det = frame_detection[detection_idxs[0]]
                            track_head_list.append(det)
                        for tail_index in tail_index_list:
                            frame_detection1 = track_item[tail_index]
                            detection_idxs1 = list(frame_detection1.keys())
                            det1 = frame_detection1[detection_idxs1[0]]
                            track_tail_list.append(det1)
                        head_track_list.append(track_head_list)
                        tail_track_list.append(track_tail_list)
                        track_is_end.append(self.tracks_state['{}'.format(track_id)])
                        # print(track_item)
                    # 判断取得的轨迹的情况
                    # start_deal用于判断是否继续处理轨迹的匹配，如果当前取出的轨迹中存在轨迹尚未结束，且后段轨迹长度小于3，则先不处理
                    # 如果后段轨迹长度大于3，或者当前轨迹已经结束，则可以继续处理
                    start_deal = True
                    for atail_track_id in range(len(tail_track_list)):
                        if len(tail_track_list[atail_track_id]) <15 and track_is_end[atail_track_id] is False:
                            # 轨迹未结束且长度小于6，下次再处理
                            start_deal = False
                    # 记录涉及不能处理的轨迹
                    tracks_temp_f = []
                    for trac_pair in self.tracks_cant_deal:
                        for a in trac_pair:
                            if a not in tracks_temp_f:
                                tracks_temp_f.append(a)
                    # 判断当前帧涉及的轨迹是否在尚不能处理的轨迹里
                    flag = False
                    for atra in tracks:
                        if atra in tracks_temp_f:
                            flag = True
                            break
                    # 判断当前轨迹对是否已经在尚不能处理的轨迹里且是最早未被处理的
                    flag1 = False
                    labbb = 0
                    for tr in tracks:
                        if labbb == 0:
                            for pair_pair in self.tracks_cant_deal:
                                if tr in pair_pair:
                                    if tracks == pair_pair:
                                        flag1 = True
                                    else:
                                        labbb = 1
                        else:
                            break
                    if (start_deal is True and flag is False) or (start_deal is True and (flag is True and flag1 is True)):
                        if flag is True and flag1 is True:
                            ind = self.tracks_cant_deal.index(tracks)
                            self.tracks_cant_deal.pop(ind)
                        # 记录当前检测最终属于哪条轨迹
                        track_cur_det_in = -1
                        temp_track_record = copy.deepcopy(self.tracks_record)
                        temp_track_state = copy.deepcopy(self.tracks_state)
                        # count作为计数器，记录前段轨迹中长度为1的个数，当前段轨迹中全部长度都为1,
                        # 需要比较后段轨迹的和谐度，从而确定当前检测属于哪个轨迹
                        count = 0
                        # 匹配状态矩阵，用于记录轨迹的匹配状态,初始化为-2
                        # 当前段轨迹长度为1时，匹配状态置为-1
                        match_state = -3*np.ones((len(head_track_list),2))
                        matched_angles_temp = []
                        for head_tracks_id in range(len(head_track_list)):
                            # 遍历前段轨迹列表
                            head_tracks = head_track_list[head_tracks_id]
                            if len(head_tracks) == 1:
                                # match_state[head_tracks_id][0] = -1
                                angles = [180. for _ in range(len(head_track_list))]
                                matched_angles_temp.append(angles)
                                count = count + 1
                            else:
                                # 前段轨迹长度不为0，可以参与匹配
                                match_angles = self.match_bet_tracks(head_tracks, tail_track_list)
                                matched_angles_temp.append(match_angles)
                        # 统一处理分配
                        matched_angles_temp = np.array(matched_angles_temp)
                        for head_tracks_id in range(len(head_track_list)):
                            head_tracks = head_track_list[head_tracks_id]
                            if len(head_tracks) == 1:
                                match_state[head_tracks_id][0] = -1
                            else:
                                row,col = self.math_operation.get_min_row_col(matched_angles_temp)
                                if matched_angles_temp[row][col] <90:
                                    match_state[row][0] = col
                                    match_state[col][1] = row
                                else:
                                    match_state[row][0] = -2
                                matched_angles_temp[row,:] = 190
                                matched_angles_temp[:,col] = 190
                        # 当前段轨迹中全部长度都为1,需要比较后段轨迹的和谐度，从而确定当前检测属于哪个轨迹
                        if count == len(head_track_list):
                            for tail_tracks_id in range(len(tail_track_list)):
                                # 由于前段轨迹长度均为1，及该共同检测，所以最匹配的轨迹直接保留
                                match_state[tail_tracks_id][0] = tail_tracks_id
                                match_state[tail_tracks_id][1] = tail_tracks_id
                                # 最匹配的轨迹直接保留,其他轨迹断开
                                # macth_angles = [180 for _ in range(len(tail_track_list))]
                            # tail_track_less_than2 = 0
                            # for tail_tracks_id in range(len(tail_track_list)):
                            #     # 取出一段后段轨迹
                            #     tail_tracks = tail_track_list[tail_tracks_id]
                            #     if len(tail_tracks) > 2:
                            #         head_part = tail_tracks[0:2]
                            #         behind_part = tail_tracks[1:]
                            #         head_pca_vector = self.cal_track_feature(head_part)
                            #         behind_pca_vector = self.cal_track_feature(behind_part)
                            #         match_rate = self.cal_match_rate(head_pca_vector, behind_pca_vector)
                            #         macth_angles[tail_tracks_id] = match_rate[0,0]
                            #         # 这种情况不应该出现，因为轨迹出现共同检测时，必定
                            #         # tail_track_less_than2 +=1
                            # min_index = np.argmin(macth_angles)
                            # if macth_angles[min_index] <= 90:
                            #     # 由于前段轨迹长度均为1，及该共同检测，所以最匹配的轨迹直接保留
                            #     match_state[min_index][0] = min_index
                            #     match_state[min_index][1] = min_index
                            #     # 最匹配的轨迹直接保留,其他轨迹断开
                            #     for aa_tracks_id in range(len(tail_track_list)):
                            #         if aa_tracks_id != min_index:
                            #             match_state[aa_tracks_id][0] = -2
                            # else:
                            #     # 没有最匹配的轨迹,轨迹全部断开
                            #     for aa_tracks_id in range(len(tail_track_list)):
                            #         match_state[aa_tracks_id][0] = -2
                        # 根据匹配状态处理轨迹的合并与更正
                        # 由于此处已经处理了轨迹匹配，需要确定当前检测的归属,由track_cur_det_in记录
                        # 这里可能是轨迹交叉，这时该检测属于两个轨迹，取出现的第一个

                        # 根据后段轨迹的匹配情况进行处理，
                        # 对于与其他轨迹匹配的，前面已经进行了处理
                        # 对于没有匹配任何前段轨迹的，状态应为-2，此时应该查看前段轨迹的情况，
                        # 前段轨迹长度为1，状态为-1,如果只有一个，直接匹配，有多个则进行以此匹配计算，如果没有-1的，直接断开
                        for row in range(match_state.shape[0]):
                            if int(match_state[row][1]) == -3:
                                # 统计前段轨迹中状态为-1的轨迹
                                head_track_sub1_list = []
                                for row_t in range(match_state.shape[0]):
                                    if int(match_state[row_t][0]) == -1:
                                        head_track_sub1_list.append(row_t)
                                if len(head_track_sub1_list) > 1:
                                    # 前段轨迹中状态为-1的轨迹数大于1,分配给第一个为-1的
                                    track_ind = head_track_sub1_list[0]
                                    # 直接将当前后段轨迹分配给该前段轨迹
                                    if row == track_ind:
                                        # 不用处理
                                        match_state[head_track_sub1_list[0]][0] = -3
                                        track_cur_det_in = head_track_sub1_list[0]
                                        continue
                                    else:
                                        # 将当前检测分配给该前段轨迹
                                        track_cur_det_in = tracks[track_ind]
                                        # 由于交换了轨迹的部分，需要对self.tracks的状态进行更新，以对应轨迹号，这里先记录
                                        matched_track_pair = [tracks[track_ind], tracks[row]]
                                        # dealed_track_reord = update_tracks(dealed_track_reord,
                                        #                                    tracks[track_ind],
                                        #                                    tracks[row])
                                        dealed_track_reord_frame.append(matched_track_pair)
                                        # 取后段轨迹
                                        tail_track1 = self.tracks_record['{}'.format(tracks[row])]
                                        a_temp_track_id = tracks[row]
                                        tail_track_sorted_index1 = sorted(tail_track1.keys(),
                                                                          key=lambda k: int(k))
                                        # 取出当前检测在轨迹中的索引位置
                                        tail_detection_posi_index1 = tail_track_sorted_index1.index(
                                            frame_id)
                                        tail_part_inds = tail_track_sorted_index1[
                                                         tail_detection_posi_index1:]
                                        head_temp = {}
                                        head_temp['{}'.format(tracks[track_ind])] = {}
                                        for ind in tail_part_inds:
                                            head_temp['{}'.format(tracks[track_ind])][ind] = \
                                            tail_track1[ind]
                                        # 将后段轨迹与前段轨迹合并
                                        temp_track_record['{}'.format(tracks[track_ind])] = head_temp[
                                            '{}'.format(tracks[track_ind])]
                                        temp_track_state['{}'.format(tracks[track_ind])] = \
                                        self.tracks_state[
                                            '{}'.format(tracks[row])]
                                        # 删除后段轨迹
                                        # temp_track_record.pop('{}'.format(tracks[row]))
                                        # temp_track_state.pop('{}'.format(matched_head_track_ind))

                                        lab = 1
                                        match_state[track_ind][0] = -3
                                elif len(head_track_sub1_list) == 1:
                                    # 直接将当前后段轨迹分配给该前段轨迹
                                    if row == head_track_sub1_list[0]:
                                        # 不用处理
                                        match_state[head_track_sub1_list[0]][0] = -3
                                        track_cur_det_in = head_track_sub1_list[0]
                                        continue
                                    else:
                                        # 将当前检测分配给该前段轨迹
                                        matched_head_track_ind = head_track_sub1_list[0]
                                        track_cur_det_in = tracks[matched_head_track_ind]
                                        # 由于交换了轨迹的部分，需要对self.tracks的状态进行更新，以对应轨迹号，这里先记录
                                        matched_track_pair = [tracks[matched_head_track_ind],
                                                              tracks[row]]
                                        # dealed_track_reord = update_tracks(dealed_track_reord, tracks[
                                        #     matched_head_track_ind], tracks[row])
                                        dealed_track_reord_frame.append(matched_track_pair)
                                        # 取后段轨迹
                                        tail_track1 = self.tracks_record['{}'.format(tracks[row])]
                                        a_temp_track_id = tracks[row]
                                        tail_track_sorted_index1 = sorted(tail_track1.keys(),
                                                                          key=lambda k: int(k))
                                        # 取出当前检测在轨迹中的索引位置
                                        tail_detection_posi_index1 = tail_track_sorted_index1.index(
                                            frame_id)
                                        tail_part_inds = tail_track_sorted_index1[
                                                         tail_detection_posi_index1:]
                                        head_temp = {}
                                        head_temp['{}'.format(tracks[matched_head_track_ind])] = {}
                                        for ind in tail_part_inds:
                                            head_temp['{}'.format(tracks[matched_head_track_ind])][
                                                ind] = tail_track1[ind]
                                        # 将后段轨迹与前段轨迹合并
                                        temp_track_record['{}'.format(tracks[matched_head_track_ind])] = \
                                        head_temp['{}'.format(tracks[matched_head_track_ind])]
                                        temp_track_state['{}'.format(tracks[matched_head_track_ind])] = \
                                        self.tracks_state['{}'.format(tracks[row])]
                                        lab = 1
                                        # 删除后段轨迹
                                        # temp_track_record.pop('{}'.format(tracks[row]))
                                        # temp_track_state.pop('{}'.format(matched_head_track_ind))
                                        match_state[head_track_sub1_list[0]][0] = -3
                                elif len(head_track_sub1_list) == 0:
                                    # 断开轨迹，重新开始轨迹号
                                    # 取后段轨迹
                                    tail_track1 = self.tracks_record['{}'.format(tracks[row])]
                                    a_temp_track_id = tracks[row]
                                    tail_track_sorted_index1 = sorted(tail_track1.keys(),
                                                                      key=lambda k: int(k))
                                    # 取出当前检测在轨迹中的索引位置
                                    tail_detection_posi_index1 = tail_track_sorted_index1.index(
                                        frame_id)
                                    tail_part_inds = tail_track_sorted_index1[
                                                     tail_detection_posi_index1:]
                                    head_temp = {}
                                    track_cur_det_in = self._next_id
                                    if self._next_id == 62:
                                        print("s")
                                    # 由于交换了轨迹的部分，需要对self.tracks的状态进行更新，以对应轨迹号，这里先记录
                                    matched_track_pair = [self._next_id, tracks[row]]
                                    # dealed_track_reord = update_tracks(dealed_track_reord,
                                    #                                    self._next_id, tracks[row])
                                    dealed_track_reord_frame.append(matched_track_pair)
                                    head_temp['{}'.format(self._next_id)] = {}
                                    for ind in tail_part_inds:
                                        head_temp['{}'.format(self._next_id)][ind] = tail_track1[ind]
                                    # 将后段轨迹与前段轨迹合并
                                    lab = 1
                                    temp_track_record['{}'.format(self._next_id)] = head_temp[
                                        '{}'.format(self._next_id)]
                                    temp_track_state['{}'.format(self._next_id)] = self.tracks_state[
                                        '{}'.format(tracks[row])]
                                    self._next_id += 1
                                    # match_state[head_track_sub1_list[0]][0] = -4
                        # 先根据前段轨迹的匹配情况进行处理，
                        # 只处理前段轨迹与后续轨迹匹配到同一轨迹的，保持不变
                        # 前段轨迹匹配到其他后续轨迹的，交换轨迹
                        # 前段轨迹没有可匹配后段轨迹的，断开
                        for row in range(match_state.shape[0]):
                            if int(match_state[row][0]) == row:
                                #说明轨迹正确，不用处理
                                track_cur_det_in = tracks[row]
                                continue
                            elif int(match_state[row][0]) != -1 and int(match_state[row][0]) != -2 and int(match_state[row][0]) != -3:
                                #说明当前轨迹不正确，需要处理
                                matched_track_pair = [tracks[row], tracks[int(match_state[row][0])]]
                                # 由于交换了轨迹的部分，需要对self.tracks的状态进行更新，以对应轨迹号，这里先记录
                                # dealed_track_reord = update_tracks(dealed_track_reord, tracks[row], tracks[int(match_state[row][0])])
                                dealed_track_reord_frame.append(matched_track_pair)
                                track_cur_det_in = tracks[row]
                                head_track = self.tracks_record['{}'.format(tracks[row])]
                                a_track_id_temp1 = tracks[row]
                                tail_track = self.tracks_record['{}'.format(tracks[int(match_state[row][0])])]
                                a_track_id_temp2 = tracks[int(match_state[row][0])]
                                head_track_sorted_index = sorted(head_track.keys(), key=lambda k: int(k))
                                # 取出当前检测在轨迹中的索引位置
                                head_detection_posi_index = head_track_sorted_index.index(frame_id)
                                tail_track_sorted_index = sorted(tail_track.keys(), key=lambda k: int(k))
                                # 取出当前检测在轨迹中的索引位置
                                tail_detection_posi_index = tail_track_sorted_index.index(frame_id)
                                head_part_inds = head_track_sorted_index[:head_detection_posi_index]
                                tail_part_inds = tail_track_sorted_index[tail_detection_posi_index:]
                                head_temp = {}
                                head_temp['{}'.format(tracks[row])] = {}
                                for ind in head_part_inds:
                                    head_temp['{}'.format(tracks[row])][ind] = head_track[ind]
                                for ind in tail_part_inds:
                                    head_temp['{}'.format(tracks[row])][ind] = tail_track[ind]
                                temp_track_record['{}'.format(tracks[row])] = head_temp['{}'.format(tracks[row])]
                                temp_track_state['{}'.format(tracks[row])] = self.tracks_state['{}'.format(tracks[int(match_state[row][0])])]
                                lab = 1
                                # dealed_track_reord = update_tracks(dealed_track_reord,origin_track,cur_track)
                                print("s")
                            elif int(match_state[row][0]) == -2:
                                # -2说明该轨迹，前段轨迹长度合理，而没有匹配的后段轨迹，此时需要将轨迹从此处断开
                                # track_cur_det_in = tracks[row]不能确定，将之归于后段轨迹
                                head_track = self.tracks_record['{}'.format(tracks[row])]
                                a_track_id_temp = tracks[row]
                                head_track_sorted_index = sorted(head_track.keys(), key=lambda k: int(k))
                                # 取出当前检测在轨迹中的索引位置
                                head_detection_posi_index = head_track_sorted_index.index(frame_id)
                                head_part_inds = head_track_sorted_index[:head_detection_posi_index]
                                head_temp = {}
                                head_temp['{}'.format(tracks[row])] = {}
                                for ind in head_part_inds:
                                    head_temp['{}'.format(tracks[row])][ind] = head_track[ind]
                                temp_track_record['{}'.format(tracks[row])] = head_temp['{}'.format(tracks[row])]
                                temp_track_state['{}'.format(tracks[row])] = True

                        # 该检测处理完毕，进行更新
                        id = frame_id
                        temp_te = track_cur_det_in
                        self.detection_record[frame_id][detection_id] = [track_cur_det_in]
                        temp_detection_dict[frame_id][detection_id] = [track_cur_det_in]
                        self.tracks_record = temp_track_record
                        self.tracks_state = temp_track_state
                    else:
                        # 记录尚不能处理的轨迹，后续涉及该轨迹的也不能处理
                        if tracks not in self.tracks_cant_deal:
                            self.tracks_cant_deal.append(tracks)

            if lab == 1:
                # 每一帧处理完毕，需要更新self.detection_records中的轨迹号
                temp_temp_detection_dict = copy.deepcopy(temp_detection_dict)
                temp_temp_detection_record = copy.deepcopy(self.detection_record)
                tempA = []
                tempAf = -1000
                for cur,ori in dealed_track_reord_frame:
                    tempA.append([cur,tempAf])
                    temp_temp_detection_dict = update_detection_record(temp_temp_detection_dict, \
                                                                   ori,
                                                                   tempAf,
                                                                   frame_id)
                    temp_temp_detection_record = update_detection_record(temp_temp_detection_record, \
                                                                     ori,
                                                                     tempAf,
                                                                     frame_id)
                    update_track_cant_deal(tempAf,ori)
                    tempAf += 1
                temp_temp_detection_dict,temp_temp_detection_record = replace(temp_temp_detection_record, temp_temp_detection_dict,tempA,frame_id)
                self.detection_record = temp_temp_detection_record
                temp_detection_dict = temp_temp_detection_dict
            dealed_track_reord.append(dealed_track_reord_frame)
            self.deal_records.append(dealed_track_reord_frame)
        return dealed_track_reord

    # 根据后处理的结果更新轨迹状态
    def update_tracks(self,dealed_track_reord):
        for dealed_track_reord_frame in dealed_track_reord:
            tracks_temp = copy.deepcopy(self.tracks)
            for head_trackid,tail_trackid in dealed_track_reord_frame:
                track_indexs = [index for index, t in enumerate(self.tracks)]
                track_ids = [t.track_id for index, t in enumerate(self.tracks)]
                # 更新self.tracks
                if tail_trackid not in track_ids:
                    continue
                if head_trackid not in track_ids:
                    # 说明是一个断开了，重新分配了轨迹号的轨迹
                    # 需要加入轨迹列表
                    tail_track_ind = track_ids.index(tail_trackid)
                    temp = copy.deepcopy(self.tracks[tail_track_ind])
                    temp.track_id = head_trackid
                    new_track = temp
                    tracks_temp.append(new_track)
                else:
                    head_track_ind = track_ids.index(head_trackid)
                    tail_track_ind = track_ids.index(tail_trackid)
                    temp1 = copy.deepcopy(self.tracks[tail_track_ind])
                    tracks_temp[head_track_ind] = temp1
                    tracks_temp[head_track_ind].track_id = head_trackid
            # 更新se
            self.tracks = tracks_temp

        temp_tracks = []
        for track in self.tracks:
            if "{}".format(track.track_id) in self.tracks_record.keys() and self.tracks_state["{}".format(track.track_id)]==False:
                temp_tracks.append(track)
        self.tracks = temp_tracks

    def post_process(self):
        # 遍历frame索引
        for frame_id in self.detection_record.keys():
            # 遍历检测索引
            for detection_id in self.detection_record[frame_id].keys():
                if len(self.detection_record[frame_id][detection_id])>1:
                    #大于1，说明发生了轨迹分裂,当前检测属于多个轨迹
                    head_track_list = []
                    tail_track_list = []
                    # 轨迹片段的长度
                    track_episode_length = 20
                    # 取轨迹
                    tracks = self.detection_record[frame_id][detection_id]
                    for track_id in self.detection_record[frame_id][detection_id]:
                        # track_item是一条轨迹，包含以帧号为键，检测号为值的字典
                        track_item = self.tracks_record['{}'.format(track_id)]
                        # 需要先对轨迹中的检测按帧顺序进行排序
                        track_sorted_index = sorted(track_item.keys(), key=lambda k: int(k))
                        # 取出当前检测在轨迹中的索引位置
                        detection_posi_index = track_sorted_index.index(frame_id)
                        head_index_list = None
                        tail_index_list = None
                        # 从当前检测在轨迹中的索引位置，取轨迹长
                        if detection_posi_index+1 >= track_episode_length:
                             head_index_list = track_sorted_index[(detection_posi_index+1)-track_episode_length:detection_posi_index+1]
                        else:
                            head_index_list = track_sorted_index[0:detection_posi_index+1]
                        if detection_posi_index+1+track_episode_length <= len(track_sorted_index):
                            tail_index_list = track_sorted_index[detection_posi_index:detection_posi_index+track_episode_length]
                        else:
                            tail_index_list = track_sorted_index[detection_posi_index:]
                        track_head_list = []
                        track_tail_list = []
                        # 根据检测索引取出检测
                        for head_index in head_index_list:
                            frame_detection = track_item[head_index]
                            detection_idxs = list(frame_detection.keys())
                            det = frame_detection[detection_idxs[0]]
                            track_head_list.append(det)
                        for tail_index in tail_index_list:
                            frame_detection1 = track_item[tail_index]
                            detection_idxs1 = list(frame_detection1.keys())
                            det1 = frame_detection1[detection_idxs1[0]]
                            track_tail_list.append(det1)
                        head_track_list.append(track_head_list)
                        tail_track_list.append(track_tail_list)
                        # print(track_item)
                    # 进行轨迹前后段匹配，
                    match_pair = self.track_similarity_match(head_track_list,tail_track_list)
                    match_track_ind = -1
                    # for pair_ind,pair in enumerate(match_pair):
                    #     if pair[0] == -1:
                    #         # 看对应的后段轨迹是否匹配
                    #         if pair[1] == -1:
                    #         else:
                    #         # pair ==-1, 说明该前段轨迹没有相匹配的后段轨迹, 需要处理，将该轨迹从当前检测处断开
                    #         self.
                    #     elif pair[0] != pair[1]:
                    #         # 第一个前段轨迹匹配,且重新匹配
                    #         match_track_ind = pair_ind
                    #     else:
                    #         # 第一个前段轨迹匹配,且不重新匹配，不处理
                    #         continue
                    # matched_trackid = tracks[match_track_ind]
                    # self.detection_record[frame_id][detection_id] = matched_trackid
                    # self.tracks_record





