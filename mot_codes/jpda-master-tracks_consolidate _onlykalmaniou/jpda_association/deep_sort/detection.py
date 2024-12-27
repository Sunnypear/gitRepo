# vim: expandtab:ts=4:sw=4
import numpy as np


class Detection(object):
    """
    This class represents a bounding box detection in a single image.

    Parameters
    ----------
    tlwh : array_like
        Bounding box in format `(x, y, w, h)`.
    confidence : float
        Detector confidence score.
    feature : array_like
        A feature vector that describes the object contained in this image.

    Attributes
    ----------
    tlwh : ndarray
        Bounding box in format `(top left x, top left y, width, height)`.
    confidence : ndarray
        Detector confidence score.
    feature : ndarray | NoneType
        A feature vector that describes the object contained in this image.

    """
    """
        tlwh:目标框左上角横纵坐标x, y; 宽w; 高h
        confindnce:目标类别置信度得分
        feature:ReID模块提取目标框的外观语义特征
    """

    def __init__(self, tlwh, confidence, feature):
        self.tlwh = np.asarray(tlwh, dtype=np.float)
        self.confidence = float(confidence)
        self.feature = np.asarray(feature, dtype=np.float32)
        self.hash_value = None

    def to_cxywh(self):
        # 将目标框坐标tlwh转换为中心点横纵坐标x，y; 宽w; 高h
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        return ret

    def to_tlbr(self):
        #将目标框坐标tlwh转换为左上角横纵坐标x，y; 右下角横纵坐标x, y
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        #将目标框坐标tlwh转换为中心点横纵坐标x，y; 宽高比a; 高h
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_bcxywh(self):
        # 将目标框坐标tlwh转换为底边中心点横纵坐标bx，by; 宽; 高;
        ret = self.tlwh.copy()
        # 中心点
        ret[0] += ret[2] / 2
        ret[1] += ret[3]
        return  ret
