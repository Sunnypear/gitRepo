# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
# from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment
from sklearn.utils.linear_assignment_ import linear_assignment
from .import kalman_filter


INFTY_COST = 1e+5

# matches_l, _, unmatched_detections = min_cost_matching(
#                 distance_metric, max_distance, tracks, detections,
#                 track_indices_l, unmatched_detections)

def min_cost_matching(
        distance_metric, max_distance, tracks, detections, track_indices=None,
        detection_indices=None):
    """Solve linear assignment problem.

    Parameters
    ----------
    distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
        The distance metric is given a list of tracks and detections as well as
        a list of N track indices and M detection indices. The metric should
        return the NxM dimensional cost matrix, where element (i, j) is the
        association cost between the i-th track in the given track indices and
        the j-th detection in the given detection_indices.
    max_distance : float
        Gating threshold. Associations with cost larger than this value are
        disregarded.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : List[int]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above).
    detection_indices : List[int]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above).

    Returns
    -------
    (List[(int, int)], List[int], List[int])
        Returns a tuple with the following three entries:
        * A list of matched track and detection indices.
        * A list of unmatched track indices.
        * A list of unmatched detection indices.

    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    # matches_l, _, unmatched_detections
    if len(detection_indices) == 0 or len(track_indices) == 0:
        return [], track_indices, detection_indices  # Nothing to match.
    # 即gate_metric那个函数内定义的函数
    '''
        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            # 轨迹ID列表
            targets = np.array([tracks[i].track_id for i in track_indices])
            # 1. 通过最近邻计算出代价矩阵 cosine distance
            cost_matrix = self.metric.distance(features, targets)
            # 2. 计算马氏距离,得到新的状态矩阵
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)
            return cost_matrix
    '''
    cost_matrix = distance_metric(
        tracks, detections, track_indices, detection_indices)
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5
    # 匈牙利算法，得到匹配对（row,col）
    # row: track  col: detection
    indices = linear_assignment(cost_matrix)
    # for row in range(len(indices)):
    #     indices[row,0] = detection_indices[indices[row,0]]
    #     indices[row, 1] = track_indices[indices[row, 1]]

    matches, unmatched_tracks, unmatched_detections = [], [], []
    for row, detection_idx in enumerate(detection_indices):
        if row not in indices[:, 0]:
            unmatched_detections.append(detection_idx)
    for col, track_idx in enumerate(track_indices):
        if col not in indices[:, 1]:
            unmatched_tracks.append(track_idx)
    for row, col in indices:
        track_idx = track_indices[col]
        detection_idx = detection_indices[row]
        if cost_matrix[row, col] > max_distance:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_idx,detection_idx))
    return matches, unmatched_tracks, unmatched_detections

'''
gated_metric, self.metric.matching_threshold, 
self.max_age, self.tracks, detections, confirmed_tracks
'''
# linear_assignment.matching_cascade(gated_metric, self.metric.matching_threshold,
#                                    self.max_age, self.tracks, detections, confirmed_tracks)

def matching_by_jpda_matrix(min_prob,tracks, detections,jpda_prob_matrix,track_indices=None, detection_indices=None):
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))
    unmatched_detections = detection_indices
    if len(detection_indices) == 0 or len(track_indices) == 0:
        return [], track_indices, detection_indices  # Nothing to match.
    # # 匈牙利算法，得到匹配对（row,col）
    # # row: track  col: detection
    # indices = linear_assignment(cost_matrix)
    jpda_prob_matrix_used = jpda_prob_matrix[1:,:]
    row_t,col_t = linear_sum_assignment(jpda_prob_matrix_used,True)
    row_tt = []
    col_tt = []
    for i in range(len(row_t)):
        if jpda_prob_matrix_used[row_t[i],col_t[i]] != 0.0:
            row_tt.append(row_t[i])
            col_tt.append(col_t[i])
    row_tt = np.array(row_tt)
    col_tt = np.array(col_tt)
    # indices_temp = np.c_[row_t[:, np.newaxis], col_t[:, np.newaxis]]
    indices = np.c_[row_tt[:, np.newaxis], col_tt[:, np.newaxis]]

    matches, unmatched_tracks, unmatched_detections = [], [], []
    for row, detection_idx in enumerate(detection_indices):
        if detection_idx not in indices[:, 0]:
            unmatched_detections.append(detection_idx)
    for col, track_idx in enumerate(track_indices):
        if track_idx not in indices[:, 1]:
            unmatched_tracks.append(track_idx)
    for row, col in indices:
        track_idx = track_indices[col]
        detection_idx = detection_indices[row]
        if jpda_prob_matrix_used[row, col] < min_prob:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_idx,detection_idx))
    return matches, unmatched_tracks, unmatched_detections



def matching_cascade(
        distance_metric, max_distance, cascade_depth, tracks, detections,jpda_prob_matrix,
        track_indices=None, detection_indices=None):
    """Run matching cascade.

    Parameters
    ----------
    distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
        The distance metric is given a list of tracks and detections as well as
        a list of N track indices and M detection indices. The metric should
        return the NxM dimensional cost matrix, where element (i, j) is the
        association cost between the i-th track in the given track indices and
        the j-th detection in the given detection indices.
    max_distance : float
        代价大于此值的关联将被忽略。
        Gating threshold. Associations with cost larger than this value are
        disregarded.
    cascade_depth: int
        The cascade depth, should be se to the maximum track age.
    tracks : List[track.Track]
    当前时间步长的预测轨迹列表。
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        当前时间步长的检测列表。
        A list of detections at the current time step.
    track_indices : Optional[List[int]]
    将“成本矩阵”中的行映射到“轨道”中的轨道的轨道索引列表(参见上面的描述)。默认为所有轨道。
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above). Defaults to all tracks.
    detection_indices : Optional[List[int]]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above). Defaults to all
        detections.

    Returns
    -------
    (List[(int, int)], List[int], List[int])
        Returns a tuple with the following three entries:
        * A list of matched track and detection indices.
        * A list of unmatched track indices.
        * A list of unmatched detection indices.

    """
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))

    unmatched_detections = detection_indices
    matches = []
    for level in range(cascade_depth):
        if len(unmatched_detections) == 0:  # No detections left
            break

        track_indices_l = [k for k in track_indices if tracks[k].time_since_update == 1 + level]
        if len(track_indices_l) == 0:  # Nothing to match at this level
            continue

        matches_l, _, unmatched_detections = \
            min_cost_matching(
                distance_metric, max_distance, tracks, detections,jpda_prob_matrix,
                track_indices_l, unmatched_detections)
        matches += matches_l
    unmatched_tracks = list(set(track_indices) - set(k for k, _ in matches))
    return matches, unmatched_tracks, unmatched_detections



# 根据通过卡尔曼滤波获得的状态分布，使代价矩阵中的不可行条目无效。
# INFTY_COST = 1e+5
# self.kf, cost_matrix, tracks, dets, track_indices,
#                 detection_indices
def gate_cost_matrix(kf, cost_matrix, tracks, detections, track_indices,
                     detection_indices, gated_cost=INFTY_COST, only_position=False):
    """Invalidate infeasible entries in cost matrix based on the state
    distributions obtained by Kalman filtering.

    Parameters
    ----------
    kf : The Kalman filter.
    cost_matrix : ndarray
        The NxM dimensional cost matrix, where N is the number of track indices
        and M is the number of detection indices, such that entry (i, j) is the
        association cost between `tracks[track_indices[i]]` and
        `detections[detection_indices[j]]`.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : List[int]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above).
    detection_indices : List[int]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above).
    gated_cost : Optional[float]
        Entries in the cost matrix corresponding to infeasible associations are
        set this value. Defaults to a very large value.
    only_position : Optional[bool]
        If True, only the x, y position of the state distribution is considered
        during gating. Defaults to False.

    Returns
    -------
    ndarray
        Returns the modified cost matrix.

    """
    gating_dim = 2 if only_position else 4
    """
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
    """
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    #
    measurements = np.asarray(
        [detections[i].to_xyah() for i in detection_indices])
    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        # 设置为inf
        cost_matrix[row, gating_distance > gating_threshold] = gated_cost
        """
        这段代码的作用是将`cost_matrix`中满足条件`gating_distance > gating_threshold`的
        行索引`row`上的对应元素值设为`gated_cost`。
        更具体地讲，`cost_matrix`是一个二维数组，`row`是一个整数变量，
        `gating_distance`是一个与`cost_matrix`相同维度的一维数组，
        `gating_threshold`是一个阈值，`gated_cost`是一个要赋给满足条件的元素的特定值。
        在该代码中，`gating_distance > gating_threshold`是一个布尔型的条件表达式，
        返回一个与`gating_distance`维度相同的布尔型数组，
        该数组中的元素值为布尔类型的`True`或`False`，表示对应位置满足或不满足条件。
       `cost_matrix[row, gating_distance > gating_threshold]`的写法
是对二维数组`cost_matrix`进行索引操作，其中行索引指定为`row`，
列索引使用布尔型数组`gating_distance > gating_threshold`的`True`值所在的位置。
这样，就选择了满足条件的行索引的对应元素。
并将其赋值为`gated_cost`，完成了对满足条件的元素的修改操作。
        """
    return cost_matrix
