# vim: expandtab:ts=4:sw=4
"""
该模块包含一个图像查看器和基于OpenCV的绘图例程。
This module contains an image viewer and drawing routines based on OpenCV.
"""
import numpy as np
import cv2
import time


def is_in_bounds(mat, roi):
    # 检查ROI是否完全包含在图像中。
    """Check if ROI is fully contained in the image.

    Parameters
    ----------
    mat : ndarray
        An ndarray of ndim>=2.
    roi : (int, int, int, int)
    感兴趣的区域(x, y，宽度，高度)，其中(x, y)是左上角。
        Region of interest (x, y, width, height) where (x, y) is the top-left
        corner.

    Returns
    -------
    bool
    Returns true if the ROI is contain in mat.
        Returns true if the ROI is contain in mat.

    """
    if roi[0] < 0 or roi[0] + roi[2] >= mat.shape[1]:
        return False
    if roi[1] < 0 or roi[1] + roi[3] >= mat.shape[0]:
        return False
    return True


def view_roi(mat, roi):
    """Get sub-array.
    ROI必须是有效的，即完全包含在图像中

    The ROI must be valid, i.e., fully contained in the image.

    Parameters
    ----------
    mat : ndarray
        An ndarray of ndim=2 or ndim=3.
    roi : (int, int, int, int)
        Region of interest (x, y, width, height) where (x, y) is the top-left
        corner.

    Returns
    -------
    ndarray
        A view of the roi.

    """
    sx, ex = roi[0], roi[0] + roi[2]
    sy, ey = roi[1], roi[1] + roi[3]
    if mat.ndim == 2:
        return mat[sy:ey, sx:ex]
    else:
        return mat[sy:ey, sx:ex, :]


class ImageViewer(object):
    """
    具有绘图例程和视频捕获功能的图像查看器。
    An image viewer with drawing routines and video capture capabilities.

    Key Bindings:

    * 'SPACE' : pause
    * 'ESC' : quit

    Parameters
    ----------
    update_ms : int
    帧之间的毫秒数(1000 /帧每秒)。
        Number of milliseconds between frames (1000 / frames per second).
    window_shape : (int, int)
    窗口的形状(宽度，高度)
        Shape of the window (width, height).
    caption : Optional[str]
        Title of the window.

    Attributes
    ----------
    image : ndarray
    形状(高度，宽度，3)的彩色图像。您可以直接操作该图像来改变视图。
    否则，您可以调用该类的任何绘图例程。在内部，图像被视为在BGR色彩空间中。
        Color image of shape (height, width, 3). You may directly manipulate
        this image to change the view. Otherwise, you may call any of the
        drawing routines of this class. Internally, the image is treated as
        beeing in BGR color space.
请注意，在可视化之前，图像被调整为图像查看器窗口形状的大小。
因此，您可以传递不同大小的图像，并使用适当的原始点坐标调用绘图例程。
        Note that the image is resized to the the image viewers window_shape
        just prior to visualization. Therefore, you may pass differently sized
        images and call drawing routines with the appropriate, original point
        coordinates.
    color : (int, int, int)
    适用于所有绘图例程的当前BGR颜色代码。取值范围为[0-255]。
        Current BGR color code that applies to all drawing routines.
        Values are in range [0-255].
    text_color : (int, int, int)
    适用于所有文本呈现例程的当前BGR文本颜色代码。取值范围为[0-255]。
        Current BGR text color code that applies to all text rendering
        routines. Values are in range [0-255].
    thickness : int
    适用于所有绘图例程的描边宽度(以像素为单位)。
        Stroke width in pixels that applies to all drawing routines.

    """

    def __init__(self, update_ms, window_shape=(640, 480), caption="Figure 1"):
        self._window_shape = window_shape
        self._caption = caption
        self._update_ms = update_ms
        self._video_writer = None
        self._user_fun = lambda: None
        self._terminate = False
        # (640, 480)+(3,) = (640, 480, 3)
        self.image = np.zeros(self._window_shape + (3, ), dtype=np.uint8)
        self._color = (0, 0, 0)
        self.text_color = (255, 255, 255)
        self.thickness = 1

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, value):
        if len(value) != 3:
            raise ValueError("color must be tuple of 3")
        self._color = tuple(int(c) for c in value)

    def rectangle(self, x, y, w, h, label=None):
        """Draw a rectangle.

        Parameters
        ----------
        x : float | int
            Top left corner of the rectangle (x-axis).
        y : float | int
            Top let corner of the rectangle (y-axis).
        w : float | int
            Width of the rectangle.
        h : float | int
            Height of the rectangle.
        label : Optional[str]
        放置在矩形左上角的文本标签。
            A text label that is placed at the top left corner of the
            rectangle.

        """
        pt1 = int(x), int(y)
        pt2 = int(x + w), int(y + h)
        cv2.rectangle(self.image, pt1, pt2, self._color, self.thickness)
        if label is not None:
            # getTextSize用于给出opencv绘制文字时，文字的绘制范围，
            # 会计算出包围这些文字的包围盒
            # text_size 为（文字包围框宽，文字包围框高），文字包围框最低点距基线高
            text_size = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_PLAIN, 1, self.thickness)

            center = pt1[0] + 5, pt1[1] + 5 + text_size[0][1]
            pt2 = pt1[0] + 10 + text_size[0][0], pt1[1] + 10 + \
                text_size[0][1]
            cv2.rectangle(self.image, pt1, pt2, self._color, -1)
            cv2.putText(self.image, label, center, cv2.FONT_HERSHEY_PLAIN,
                        1, (255, 255, 255), self.thickness)

    def circle(self, x, y, radius, label=None):
        """Draw a circle.

        Parameters
        ----------
        x : float | int
            Center of the circle (x-axis).
        y : float | int
            Center of the circle (y-axis).
        radius : float | int
            Radius of the circle in pixels.
        label : Optional[str]
            A text label that is placed at the center of the circle.

        """
        image_size = int(radius + self.thickness + 1.5)  # actually half size
        roi = int(x - image_size), int(y - image_size), \
            int(2 * image_size), int(2 * image_size)
        if not is_in_bounds(self.image, roi):
            return

        image = view_roi(self.image, roi)
        center = image.shape[1] // 2, image.shape[0] // 2
        cv2.circle(
            image, center, int(radius + .5), self._color, self.thickness)
        if label is not None:
            cv2.putText(
                self.image, label, center, cv2.FONT_HERSHEY_PLAIN,
                2, self.text_color, 2)

    def gaussian(self, mean, covariance, label=None):
        """
        绘制二维高斯分布的95%置信椭圆。
        Draw 95% confidence ellipse of a 2-D Gaussian distribution.

        Parameters
        ----------
        mean : array_like
            高斯分布的平均向量(ndim=1)。
            The mean vector of the Gaussian distribution (ndim=1).
        covariance : array_like
            高斯分布的2x2协方差矩阵。
            The 2x2 covariance matrix of the Gaussian distribution.
        label : Optional[str]
            放置在椭圆中心的文本标签。
            A text label that is placed at the center of the ellipse.

        """
        # chi2inv(0.95, 2) = 5.9915
        """
        np.linalg.eig(a)
        求方阵(n x n)的特征值与右特征向量
        a : (…, M, M) array
        a是一个矩阵Matrix的数组。每个矩阵M都会被计算其特征值与特征向量。
        return w : (…, M) array
        返回的w是其特征值。特征值不会特意进行排序。
        返回的array一般都是复数形式，除非虚部为0，会被cast为实数。
        当a是实数类型时，返回的就是实数。
        v : (…, M, M) array
        返回的v是归一化后的特征向量（length为1）。
        特征向量v[:,i]对应特征值w[i]。
        需要说明的是，特征向量之间可能存在线性相关关系，即返回的v可能不是满秩的。
        但如果特征值都不同的话，理论上来说，所有特征向量都是线性无关的。
        此时可以利用inv(v)@ a @ v来计算特征值的对角矩阵（对角线上的元素是特征值，其余元素为0),
        同时可以用v @ diag(w) @ inv(v)来恢复a。
        同时需要说明的是，这里得到的特征向量都是右特征向量。即A x = λ x 
        """
        vals, vecs = np.linalg.eigh(5.9915 * covariance)
        # vals.argsort()返回从小到大的索引
        # 使用[::-1], 可以建立X从大到小的索引。
        indices = vals.argsort()[::-1]
        vals, vecs = np.sqrt(vals[indices]), vecs[:, indices]

        """
        这段代码定义了一个名为`gaussian`的方法，用于绘制二维高斯分布的95%置信椭圆。
        方法需要传入三个参数：
        1. `mean`：高斯分布的平均向量，用来确定椭圆的中心位置。
        2. `covariance`：高斯分布的2x2协方差矩阵，用来确定椭圆的形状。
        3. `label`：可选参数，是一个放置在椭圆中心的文本标签。
        1. 使用`np.linalg.eigh`函数计算协方差矩阵的特征值和特征向量。
           - `vals`数组中的特征值按从小到大的顺序排列。
           - `vecs`数组中的每一列是相应特征值的特征向量。
        2. 对特征值进行排序和平方根运算，以及对特征向量进行调整和取舍。
           - 使用`argsort()`函数对特征值的索引进行排序，返回从小到大的索引数组`indices`。
           - 使用`[::-1]`来反转索引数组的顺序，从而得到特征值从大到小的索引。
           - 根据排序后的索引，将特征值和特征向量重新排序和选择。
           - 对特征值取平方根，得到椭圆的主轴长度。
        3. 计算椭圆的中心、主轴和旋转角度。
           - 根据平均向量的第一个和第二个元素，取整并加上0.5，得到椭圆的中心坐标。
           - 根据主轴长度的第一个和第二个元素，取整并加上0.5，得到椭圆的主轴长度。
           - 通过特征向量计算旋转角度。
        4. 使用OpenCV的`cv2.ellipse`函数绘制椭圆。
           - 将椭圆的中心、主轴、旋转角度等参数传递给`cv2.ellipse`函数，绘制椭圆。
5. 如果传入了标签参数，使用OpenCV的`cv2.putText`函数在椭圆中心位置绘制文本标签。
        """
        center = int(mean[0] + .5), int(mean[1] + .5)
        axes = int(vals[0] + .5), int(vals[1] + .5)
        angle = int(180. * np.arctan2(vecs[1, 0], vecs[0, 0]) / np.pi)
        cv2.ellipse(
            self.image, center, axes, angle, 0, 360, self._color, 2)
        if label is not None:
            cv2.putText(self.image, label, center, cv2.FONT_HERSHEY_PLAIN,
                        2, self.text_color, 2)

    def annotate(self, x, y, text):
        """Draws a text string at a given location.

        Parameters
        ----------
        x : int | float
            Bottom-left corner of the text in the image (x-axis).
        y : int | float
            Bottom-left corner of the text in the image (y-axis).
        text : str
            The text to be drawn.

        """
        cv2.putText(self.image, text, (int(x), int(y)), cv2.FONT_HERSHEY_PLAIN,
                    2, self.text_color, 2)

    def colored_points(self, points, colors=None, skip_index_check=False):
        """Draw a collection of points.
        点的大小固定为1。
        The point size is fixed to 1.
        Parameters
        ----------
        points : ndarray
        图像位置的Nx2数组，其中第一维为x坐标，第二维为y坐标。
            The Nx2 array of image locations, where the first dimension is
            the x-coordinate and the second dimension is the y-coordinate.
        colors : Optional[ndarray]
        Nx3的颜色数组(dtype=np.uint8)。如果为None，则使用当前颜色属性。
            The Nx3 array of colors (dtype=np.uint8). If None, the current
            color attribute is used.
        skip_index_check : Optional[bool]
        如果为True，则跳过索引范围检查。这样更快，但要求所有点都在图像尺寸内。
            If True, index range checks are skipped. This is faster, but
            requires all points to lie within the image dimensions.

        """
        if not skip_index_check:
            cond1, cond2 = points[:, 0] >= 0, points[:, 0] < 480
            cond3, cond4 = points[:, 1] >= 0, points[:, 1] < 640
            indices = np.logical_and.reduce((cond1, cond2, cond3, cond4))
            points = points[indices, :]
        if colors is None:
            colors = np.repeat(
                self._color, len(points)).reshape(3, len(points)).T
        indices = (points + .5).astype(np.int)
        self.image[indices[:, 1], indices[:, 0], :] = colors

    def enable_videowriter(self, output_filename, fourcc_string="MJPG",
                           fps=None):
        """ Write images to video file.

        Parameters
        ----------
        output_filename : str
            Output filename.
        fourcc_string : str
            The OpenCV FOURCC code that defines the video codec (check OpenCV
            documentation for more information).
        fps : Optional[float]
            Frames per second. If None, configured according to current
            parameters.

        """
        fourcc = cv2.VideoWriter_fourcc(*fourcc_string)
        if fps is None:
            fps = int(1000. / self._update_ms)
        self._video_writer = cv2.VideoWriter(
            output_filename, fourcc, fps, self._window_shape)

    def disable_videowriter(self):
        """ Disable writing videos.
        """
        self._video_writer = None

    def deal(self, update_fun=None):
        """Start the image viewer.
        此方法阻塞，直到用户请求关闭窗口。
        This method blocks until the user requests to close the window.
        Parameters
        ----------
        update_fun : Optional[Callable[] -> None]
        一个可选的可调用对象，在每一帧被调用。可用于播放动画/视频序列。
            An optional callable that is invoked at each frame. May be used
            to play an animation/a video sequence.
        """
        # self._user_fun = lambda: None
        if update_fun is not None:
            self._user_fun = update_fun

        self._terminate, is_paused = False, False
        # print("ImageViewer is paused, press space to start.")
        while not self._terminate:
            if not is_paused:
                # 在此处循环执行回调函数
                self._terminate = not self._user_fun()


    def run(self, update_fun=None):
        """Start the image viewer.
        此方法阻塞，直到用户请求关闭窗口。
        This method blocks until the user requests to close the window.
        Parameters
        ----------
        update_fun : Optional[Callable[] -> None]
        一个可选的可调用对象，在每一帧被调用。可用于播放动画/视频序列。
            An optional callable that is invoked at each frame. May be used
            to play an animation/a video sequence.

        """
        # self._user_fun = lambda: None
        if update_fun is not None:
            self._user_fun = update_fun

        self._terminate, is_paused = False, False
        # print("ImageViewer is paused, press space to start.")
        while not self._terminate:
            t0 = time.time()
            if not is_paused:
                # 在此处循环执行回调函数
                self._terminate = not self._user_fun()
                if self._video_writer is not None:
                    self._video_writer.write(
                        cv2.resize(self.image, self._window_shape))
            t1 = time.time()
            remaining_time = max(1, int(self._update_ms - 1e3*(t1-t0)))
            # (640, 480)+(3,) = (640, 480, 3)
            # self.image = np.zeros(self._window_shape + c, dtype=np.uint8)
            # window_shape = (640, 480)
            '''
                # Update visualization.
                if display:
                    image = cv2.imread(
                        seq_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR)
                    vis.set_image(image.copy())
                    vis.draw_detections(detections)
                    vis.draw_trackers(tracker.tracks)
        
                # Store results.
                for track in tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue
                    bbox = track.to_tlwh()
                    results.append([
                        frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])

            '''

            cv2.imshow(self._caption, cv2.resize(self.image, self._window_shape[:2]))
            key = cv2.waitKey(remaining_time)
            if key & 255 == 27:  # ESC
                print("terminating")
                self._terminate = True
            elif key & 255 == 32:  # ' '
                print("toggeling pause: " + str(not is_paused))
                is_paused = not is_paused
            elif key & 255 == 115:  # 's'
                print("stepping")
                self._terminate = not self._user_fun()
                is_paused = True

        # Due to a bug in OpenCV we must call imshow after destroying the
        # window. This will make the window appear again as soon as waitKey
        # is called.
        #
        # see https://github.com/Itseez/opencv/issues/4535
        self.image[:] = 0
        cv2.destroyWindow(self._caption)
        cv2.waitKey(1)
        cv2.imshow(self._caption, self.image)

    def stop(self):
        """Stop the control loop.

        After calling this method, the viewer will stop execution before the
        next frame and hand over control flow to the user.

        Parameters
        ----------

        """
        self._terminate = True
