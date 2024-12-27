# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
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
    def __init__(self, metric, max_iou_distance=0.7, max_age=30, n_init=3,dynamic_model_list=['constant_velocity'],model_max_velocity=[8.0]):
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
        self.unmatched_detetctions = []
        self.unmatched_detetctions_frame_id = -1
        # 模型最大速度
        self.model_max_velocity = model_max_velocity
        self.jpda_prob_calculator = Joint_prob_data_associate()

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

    # 遍历每个track都进行一次预测
    def start_new_track(self, detections):
        for detection in self.unmatched_detetctions:
            self._initiate_track(detection, detections, self.unmatched_detetctions_frame_id)
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
            track.find_detections_in_association_Doors(detections,frame_idx,threshold)
        if len(self.tracks)>0:
            self.jpda_prob_calculator.update_tracks_and_detections(self.tracks,detections,threshold)
            # 根据每个track的关联域生成track与detections的关联矩阵
            self.jpda_prob_calculator.generate_associate_matrix(len(self.dynamic_model_list))
            # 生成联合事件（连通图）
            associate_events = self.jpda_prob_calculator.generate_associate_events()
            # 联合概率矩阵
            jpda_prob_matrix = np.zeros((len(detections)+1,len(self.tracks)),dtype=np.float64)

            for a_associate_event in associate_events:
                if len(a_associate_event['tracks'])!=0 and len(a_associate_event['detections'])!=0:
                    # if len(a_associate_event['tracks']) ==1 and len(a_associate_event['detections'])==1:
                    #     continue
                    # else:
                    associate_events_jpda_prob_matrix = np.zeros((len(detections) + 1, len(self.tracks)), dtype=np.float64)
                    # 对这个联合事件生成可行事件列表
                    associate_events_detections,associate_events_tracks,associate_event_matrix_list = \
                        self.jpda_prob_calculator.generate_associate_event_matrixs(a_associate_event['tracks'], a_associate_event['detections'])
                    #为当前联合事件中的可行事件计算beta(jt)
                    jpda_prob_matrix += self.jpda_prob_calculator.cal_beta_of_cur_associate_event(associate_events_jpda_prob_matrix,\
                                                     associate_events_detections,associate_events_tracks,associate_event_matrix_list)
            return jpda_prob_matrix
        else:
            return None



    # 进行测量的更新和轨迹管理
    def update(self,jpda_prob_matrix,detections,next_frame_detections,frame_idx):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        matches_a,matches_b, unmatched_tracks, unmatched_detections = \
            self._match(jpda_prob_matrix,detections)
        # 应该首先进行关联事件计算,每个轨迹进行计算，每个轨迹应该存储在关联门内的每个量测
        # for track_idx,track in enumerate(self.tracks):
        #     track.update(jpda_prob_matrix,detections,track_idx,frame_idx)
        # 1. 针对匹配上的结果
        # Update track set.
        # for track_idx, detection_idx in matches:
        #     self.tracks[track_idx].update(
        #         self.kf, detections[detection_idx])
        for track_idx, detection_idx in matches_a:
            self.tracks[track_idx].update_by_jpda(jpda_prob_matrix,detections,track_idx,detection_idx,frame_idx)
        for track_idx, detection_idx in matches_b:
            self.tracks[track_idx].update_by_kalman(self.kf, detections[detection_idx])
        for track_idx in unmatched_tracks:
            # 根据关联结果改变轨迹状态
            # self.tracks[track_idx].update_on_state_vector(self.kf,track_idx,frame_idx)
            # self.tracks[track_idx].update_by_presumed_measurement(track_idx, frame_idx)
            self.tracks[track_idx].mark_missed()
        #
        # 从这里开始初始化轨迹
        self.unmatched_detetctions = []
        self.unmatched_detetctions_frame_id = frame_idx
        for detection_idx in unmatched_detections:
            self.unmatched_detetctions.append(detections[detection_idx])
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
        matches_a, unmatched_tracks_a, unmatched_detections = \
             linear_assignment.matching_by_jpda_matrix(self.metric.matching_threshold, \
                                                       self.tracks, detections, jpda_prob_matrix, track_inds)

        # Associate remaining tracks together with unconfirmed tracks
        # using IOU.
        iou_track_candidates = [
            k for k in unmatched_tracks_a]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates)
        matches = matches_a + matches_b
        # unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches_a,matches_b,unmatched_tracks_b, unmatched_detections

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


    def _initiate_track(self, detection,next_frame_detections,frame_id):
        if next_frame_detections is None:
            dv = self.model_max_velocity[0]
            dmean = [dv, dv, 0, 0]
        else:
            next_frame_max_iou_detection = self.get_dv_by_iou_distance(detection,next_frame_detections)
            if next_frame_max_iou_detection == detection:
                print("kkkkkkkkkkkk")
                dv = self.model_max_velocity[0]
                dmean = [dv, dv, 0, 0]
            else:
                dmean = next_frame_max_iou_detection.to_xyah()-detection.to_xyah()
                sita = self.math_operation.get_track_direction_sita(next_frame_max_iou_detection.to_bcxywh()[:2],detection.to_bcxywh()[:2])
                dmean = [dmean[0], dmean[1], 0, 0]

        imm_controller = Imm_Controller(self.dynamic_model_list,self.model_max_velocity,\
                                                       deviation_process_noise = 0.5,deviation_measurement_noise=7,\
                                                       dynamic_model_transition_pro_matrix=[[1.0]])
        mean, covariance = imm_controller.initiate_dynamic_model(detection.to_xyah(),dmean)

        new_track = Track(frame_id, mean, covariance, sita,imm_controller, \
              self._next_id, self.n_init, self.max_age,
              detection.feature, detect_prob=0.9)
        self.tracks.append(new_track)
        self._next_id += 1
