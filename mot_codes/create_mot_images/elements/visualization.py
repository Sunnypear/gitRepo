# vim: expandtab:ts=4:sw=4
import numpy as np
import colorsys
from .image_viewer import ImageViewer


def create_unique_color_float(tag, hue_step=0.41):
    #为给定的轨道id(标签)创建唯一的RGB颜色代码。
    """Create a unique RGB color code for a given track id (tag).
    颜色代码是在HSV色彩空间中通过沿着色相角度移动并逐渐改变饱和度来生成的。
    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
    唯一的目标识别标签。
        The unique target identifying tag.
    hue_step : float
    HSV空间中两个相邻颜色码之间的差异(更具体地说，是色相通道中的距离)。
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (float, float, float)
        RGB color code in range [0, 1]

    """
    h, v = (tag * hue_step) % 1, 1. - (int(tag * hue_step) % 4) / 5.
    r, g, b = colorsys.hsv_to_rgb(h, 1., v)
    return r, g, b


def create_unique_color_uchar(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).
    为给定的轨道id(标签)创建唯一的RGB颜色代码。
    颜色代码是在HSV色彩空间中通过沿着色相角度移动并逐渐改变饱和度来生成的。
    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (int, int, int)
        RGB color code in range [0, 255]

    """
    r, g, b = create_unique_color_float(tag, hue_step)
    return int(255*r), int(255*g), int(255*b)


class NoVisualization(object):
    """
    一个虚拟的可视化对象，循环遍历给定序列中的所有帧以更新跟踪器，而不执行任何可视化。
    A dummy visualization object that loops through all frames in a given
    sequence to update the tracker without performing any visualization.
    """

    def __init__(self, seq_info):
        self.frame_idx = seq_info["min_frame_idx"]
        self.last_idx = seq_info["max_frame_idx"]

    def set_image(self, image):
        pass

    def draw_groundtruth(self, track_ids, boxes):
        pass

    def draw_detections(self, detections):
        pass

    def draw_trackers(self, trackers):
        pass

    def run(self, frame_callback):
        while self.frame_idx <= self.last_idx:
            frame_callback(self, self.frame_idx)
            self.frame_idx += 1


class Visualization(object):
    """
    This class shows tracking output in an OpenCV image viewer.
    """
    # [::-1]： 代表从全列表倒序取
    def __init__(self, Frames, sequence_name,update_ms):
        # image_size(576.1024): 高、宽
        image_shape = Frames[0].img.shape[:2][::-1]
        # 宽高比
        aspect_ratio = float(image_shape[1]) / image_shape[0]
        # 设置显示结果画布大小，即将图像进行缩放
        image_shape = 640, int(aspect_ratio * 640)
        # 创建ImageViewer类
        self.viewer = ImageViewer(
            update_ms, image_shape, "Figure %s" % sequence_name)
        self.viewer.thickness = 1
        self.frame_idx = 0
        self.last_idx = len(Frames)-1

    def run(self, frame_callback_show):
        """
        这行代码使用了一个 lambda 表达式来调用名为 `_update_fun` 的函数，
        并将其作为参数传递给 `run` 函数。
        lambda 表达式是一种匿名函数，用于创建简单的函数对象。
        在这个例子中，lambda 表达式用于定义一个没有参数的匿名函数，并立即调用它。
        lambda 表达式的语法形式为 `lambda arguments: expression`，
        其中 `arguments` 是函数的参数列表，`expression` 是函数的返回值表达式。
        `run` 函数将 lambda 表达式作为参数传递，并在函数内部执行它。
        这样做的好处是可以在不定义具名函数的情况下，直接传递并执行一段简短的代码逻辑。
        总结起来，这行代码的作用是调用一个匿名函数 `_update_fun(frame_callback)`，
        并将其作为参数传递给 `run` 函数进行执行。
        """
        # 此处lambda似乎是因为view.run()接受的参数是函数
        self.viewer.run(lambda: self._update_fun(frame_callback_show))

    def _update_fun(self, frame_callback):
        if self.frame_idx > self.last_idx:
            return False  # Terminate
        frame_callback(self, self.frame_idx)
        self.frame_idx += 1
        return True

    def set_image(self, image):
        self.viewer.image = image

    def draw_groundtruth(self, track_ids, boxes):
        self.viewer.thickness = 2
        for track_id, box in zip(track_ids, boxes):
            self.viewer.color = create_unique_color_uchar(track_id)
            self.viewer.rectangle(*box.astype(np.int), label=str(track_id))

    def draw_detections(self, detections):
        self.viewer.thickness = 2
        self.viewer.color = 0, 0, 255
        for i, detection in enumerate(detections):
            # 1. 单星号 * 解析可变参数
            # 传递参数前面的*，和不带星号的使用，func()是一个接收可变参数的函数。
            # func(*[1, 2, 3])  ==  func(1, 2, 3)
            # func(**{'name':'Spade_', 'number':'888888'}) == func(name='Spade_', number='888888')
            # self.viewer.rectangle(*detection.tlwh)
            self.viewer.circle(detection.tlwh[0], detection.tlwh[1], detection.tlwh[2]/2)

    def draw_trackers_by_tracks_records(self,tracks_record,frame_id):
        self.viewer.thickness = 1
        for track_id in tracks_record.keys():
            if "{}".format(frame_id) in tracks_record[track_id].keys():
                keys = list(tracks_record[track_id]["{}".format(frame_id)].keys())
                det = tracks_record[track_id]["{}".format(frame_id)][keys[0]]
                self.viewer.color = create_unique_color_uchar(int(track_id))
                self.viewer.rectangle(
                    *det.tlwh.astype(np.int), label=str(track_id))

    def draw_trackers(self, tracks):
        self.viewer.thickness = 2
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue
            self.viewer.color = create_unique_color_uchar(track.track_id)
            self.viewer.rectangle(
                *track.to_tlwh().astype(np.int), label=str(track.track_id))
            # self.viewer.gaussian(track.mean[:2], track.covariance[:2, :2],
            #                      label="%d" % track.track_id)
#
    def write_to_img(self,sequence_dir,filename):
        self.viewer.save_to_img(sequence_dir,filename)