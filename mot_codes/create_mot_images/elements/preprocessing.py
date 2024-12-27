# vim: expandtab:ts=4:sw=4
import numpy as np
import cv2


def non_max_suppression(boxes, max_bbox_overlap, scores=None):
    """
    抑制重叠检测。
    Suppress overlapping detections.
    来自[1]的原始代码已被改编以包括置信度评分
    Original code from [1]_ has been adapted to include confidence score.

    .. [1] http://www.pyimagesearch.com/2015/02/16/
           faster-non-maximum-suppression-python/

    Examples
    --------

        >>> boxes = [d.roi for d in detections]
        >>> scores = [d.confidence for d in detections]
        >>> indices = non_max_suppression(boxes, max_bbox_overlap, scores)
        >>> detections = [detections[i] for i in indices]

    Parameters
    ----------
    boxes : ndarray
        Array of ROIs (x, y, width, height).
    max_bbox_overlap : float
    重叠超过该值的roi将被抑制。
        ROIs that overlap more than this values are suppressed.
    scores : Optional[array_like]
        Detector confidence score.

    Returns
    -------
    List[int]
    返回在非极大值抑制下幸存的检测的指数。
        Returns indices of detections that have survived non-maxima suppression.

    """
    if len(boxes) == 0:
        return []

    boxes = boxes.astype(np.float)
    pick = []

    # x1： 取x
    x1 = boxes[:, 0]
    # y1: 取y
    y1 = boxes[:, 1]
    # 取右上角x坐边
    x2 = boxes[:, 2] + boxes[:, 0]
    # 取右下角y坐标
    y2 = boxes[:, 3] + boxes[:, 1]
    # 取边框面积
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    # 从小到大的索引值
    # argsort函数返回的是数组值从小到大的索引值
    if scores is not None:
        idxs = np.argsort(scores)
    else:
        idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        # i是索引
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]
        # np.where(overlap > max_bbox_overlap)[0]取0是因为返回是一个元组，overlap是矩阵类型，
        # 返回的第一位元素是行坐标索引，第二位元素是列坐标索引；对于一维矩阵，直接返回索引，第二个元素为空
        # numpy.delete(arr, obj, axis=None)
        # arr: 输入向量
        # obj: 表明哪一个子向量应该被移除。可以为整数或一个int型的向量
        # axis: 表明删除哪个轴的子向量，若默认，则返回一个被拉平的向量
        temp = np.where(overlap > max_bbox_overlap)
        temp1 = np.concatenate(([last], temp[0]))
        idxs = np.delete(idxs, temp1)

    return pick
