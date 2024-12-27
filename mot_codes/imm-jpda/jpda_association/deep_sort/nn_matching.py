  # vim: expandtab:ts=4:sw=4
import numpy as np


def _pdist(a, b):
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
    #a2[:,None]: N1   b2[None:]: 1L
    # (a[i]-b[i])**2
    # 求每个embedding的平方和
    # sum(N) + sum(L) -2 x [NxM]x[MxL] = [NxL]
    r2 = -2. * np.dot(a, b.T) + a2[:, None] + b2[None, :]
    # np.inf :+oo
    r2 = np.clip(r2, 0., float(np.inf))
    return r2


def _cosine_distance(a, b, data_is_normalized=False):
    """Compute pair-wise cosine distance between points in `a` and `b`.

    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.
    data_is_normalized : Optional[bool]
    如果为True，则假设a和b中的行是单位长度向量。否则，a和b显式归一化为长度为1
        If True, assumes rows in a and b are unit length vectors.
        Otherwise, a and b are explicitly normalized to lenght 1.

    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that eleement (i, j)
        contains the squared distance between `a[i]` and `b[j]`.

    """
    # a和b之间的余弦距离
    # a : [NxM] b : [LxM]
    # 余弦距离 = 1 - 余弦相似度
    # https://blog.csdn.net/u013749540/article/details/51813922

    if not data_is_normalized:
        # 需要将余弦相似度转化成类似欧氏距离的余弦距离。
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
        #  np.linalg.norm 操作是求向量的范式，默认是L2范式，等同于求向量的欧式距离
    return 1. - np.dot(a, b.T)
# 以上代码对应公式，注意余弦距离 = 1 - 余弦相似度。


def _nn_euclidean_distance(x, y):
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
    distances = _pdist(x, y)
    return np.maximum(0.0, distances.min(axis=0))


def _nn_cosine_distance(x, y):
    """ Helper function for nearest neighbor distance metric (cosine).

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
        smallest cosine distance to a sample in `x`.

    """
    distances = _cosine_distance(x, y)
    # 当axis=0时会分别取每一列的最大值或最小值，axis=1时，会分别取每一行的最大值或最小值
    return distances.min(axis=0)


class NearestNeighborDistanceMetric(object):
    """
    最近邻距离度量，对于每个目标，返回到目前为止观察到的任何样本的最近距离。
    A nearest neighbor distance metric that, for each target, returns
    the closest distance to any sample that has been observed so far.

    Parameters
    ----------
    metric : str
        Either "euclidean" or "cosine".
    matching_threshold: float
    匹配阈值。距离较大的样本被认为是无效匹配。
        The matching threshold. Samples with larger distance are considered an
        invalid match.
    budget : Optional[int]
    如果不是None，则将每个类的样本最多固定为这个数字。当达到预算时，删除最旧的样本。
        If not None, fix samples per class to at most this number. Removes
        the oldest samples when the budget is reached.

    Attributes
    ----------
    samples : Dict[int -> List[ndarray]]
    将目标标识映射到迄今为止观察到的样本列表的字典。
        A dictionary that maps from target identities to the list of samples
        that have been observed so far.

    """

    def __init__(self, metric, matching_threshold, budget=None):
        # 默认matching_threshold = 0.2 budge = 100
        if metric == "euclidean":
            # 使用最近邻欧氏距离
            self._metric = _nn_euclidean_distance
        elif metric == "cosine":
            # 使用最近邻余弦距离
            self._metric = _nn_cosine_distance
        else:
            raise ValueError(
                "Invalid metric; must be either 'euclidean' or 'cosine'")
        self.matching_threshold = matching_threshold
        # 在级联匹配的函数中调用
        self.budget = budget
        # budge 预算，控制feature的多少
        self.samples = {}
        # samples是一个字典{id->feature list}

    def partial_fit(self, features, targets, active_targets):
        """Update the distance metric with new data.

        Parameters
        ----------
        features : ndarray
        一个N个特征的NxM矩阵，维数为M
        确认态轨迹的外观语义特征矩阵
            An NxM matrix of N features of dimensionality M.
        targets : ndarray
        关联目标标识的整数数组。外观语义特征对应的确认态轨迹ID集合
            An integer array of associated target identities.
        active_targets : List[int]
        当前出现在场景中的目标列表。
            A list of targets that are currently present in the scene.
        active_targets:确定态轨迹ID集合
        """
        """
        list_1 = [1, 2, 3, 4]
        list_2 = ['a', 'b', 'c']
        for x, y in zip(list_1, list_2):
        并行遍历，for x in list_1 for y in list_2
        """
        # 作用：部分拟合，用新的数据更新测量距离
        # 调用：在特征集更新模块部分调用，tracker.update()中
        #
        for feature, target in zip(features, targets):
            self.samples.setdefault(target, []).append(feature)
            # 对应目标下添加新的feature，更新feature集合
            # 目标id  :  feature list
            if self.budget is not None:
                # 根据容量设置，剔除最早的外观语义特征
                self.samples[target] = self.samples[target][-self.budget:]
        # 设置预算，每个轨迹最多保留多少个特征
        # 更新确认态轨迹(同一ID)以往所有时刻的外观语义特征字典
        # 筛选激活的目标, samples里应该包括激活的和不确定的
        self.samples = {k: self.samples[k] for k in active_targets}

    def distance(self, features, targets):
        """Compute distance between features and targets.

        Parameters
        ----------
        features : ndarray
            An NxM matrix of N features of dimensionality M.
        targets : List[int]
            A list of targets to match the given `features` against.

        Returns
        -------
        ndarray
        返回形状为len(targets)， len(features)的代价矩阵，其中元素(i, j)包含
        ' targets[i] '和' features[j] '之间最接近的平方距离。
            Returns a cost matrix of shape len(targets), len(features), where
            element (i, j) contains the closest squared distance between
            `targets[i]` and `features[j]`.

        """
        # 作用：比较feature和targets之间的距离，返回一个代价矩阵
        # 调用：在匹配阶段，将distance封装为gated_metric,
        #       进行外观信息(reid得到的深度特征)+
        #       运动信息(马氏距离用于度量两个分布相似程度)
        cost_matrix = np.zeros((len(targets), len(features)))
        for i, target in enumerate(targets):
            # 一个track保存了多个特征
            cost_matrix[i, :] = self._metric(self.samples[target], features)
        return cost_matrix
