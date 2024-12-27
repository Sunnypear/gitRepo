# vim: expandtab:ts=4:sw=4
from __future__ import division, print_function, absolute_import

import argparse
import os

import cv2

import numpy as np

from application_util import preprocessing
from application_util import visualization
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
def gather_sequence_info(sequence_dir, detection_file):
    """Gather sequence information, such as image filenames, detections,
    groundtruth (if available).

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detection file.

    Returns
    -------
    Dict
        A dictionary of the following sequence information:

        * sequence_name: Name of the sequence
        * image_filenames: A dictionary that maps frame indices to image
          filenames.
        * detections: A numpy array of detections in MOTChallenge format.
        * groundtruth: A numpy array of ground truth in MOTChallenge format.
        * image_size: Image size (height, width).
        * min_frame_idx: Index of the first frame.
        * max_frame_idx: Index of the last frame.

    """
    # --sequence_dir =./ MOT16 / test / MOT16 - 06
    # --detection_file =./resources/detections/MOT16_POI_test/MOT16-06.npy
    # - -min_confidence = 0.3
    # - -nn_budget = 100
    # - -display = True

    """
    path = '/usr/local/bin/python.exe'
    filename, ext = os.path.splitext(path)
    print('文件名:', filename)
    print('扩展名:', ext)
    文件名: /usr/local/bin/python
    扩展名: .exe
    """
    """
    os.listdir()
    返回指定路径下的文件和文件夹列表
    """

    # --sequence_dir =./ MOT16 / test / MOT16 - 06
    # 读入图片列表
    image_dir = os.path.join(sequence_dir, "img1")
    #建立了一个图片索引字典 图片名前缀： 图片路径
    image_filenames = {
        int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
        # os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。
        for f in os.listdir(image_dir)}
    # 不一定有gt???
    groundtruth_file = os.path.join(sequence_dir, "gt/gt.txt")
    # --detection_file =./resources/detections/MOT16_POI_test/MOT16-06.npy
    # 读入检测数据
    detections = None
    if detection_file is not None:
        # np.load是读取npy二进制文件的
        detections = np.load(detection_file)
    groundtruth = None
    if os.path.exists(groundtruth_file):
        groundtruth = np.loadtxt(groundtruth_file, delimiter=',')

    #读入一张图片，并灰度化
    if len(image_filenames) > 0:
        """
        该行代码的作用是从一个字典
        （image_filenames）中获取第一个值（value）。
        1. image_filenames 是一个字典对象，它包含了一组键值对，
        其中键（key）是图像的标识符，而值（value）是图像的文件名。
        2. image_filenames.values() 返回一个包含所有值（即文件名）的迭代器
        （iterator）对象。
        3. iter() 函数将迭代器对象转换为一个迭代器，使其可以被逐个访问。
        4. next() 函数用于从迭代器中获取下一个元素，由于在此代码中没有指定参数，
        默认获取第一个元素。
        5. 因此，next(iter(image_filenames.values())) 
        将返回字典中的第一个文件名（即第一个值），即图像的文件名。
        """
        image = cv2.imread(next(iter(image_filenames.values())),
                           cv2.IMREAD_GRAYSCALE)
        image_size = image.shape
    else:
        image_size = None
    # 判断是否度入了文件图片
    if len(image_filenames) > 0:
        min_frame_idx = min(image_filenames.keys())
        max_frame_idx = max(image_filenames.keys())
    else:
        min_frame_idx = int(detections[:, 0].min())
        max_frame_idx = int(detections[:, 0].max())
    # --sequence_dir =./ MOT16 / test / MOT16 - 06
    #生成
    """
    {'name': 'MOT16-06', 'imDir': 'img1', 'frameRate': '14', 
    'seqLength': '1194', 'imWidth': '640', 'imHeight': '480', 'imExt': '.jpg'}
    """
    info_filename = os  .path.join(sequence_dir, "seqinfo.ini")
    if os.path.exists(info_filename):
        with open(info_filename, "r") as f:
            # line_splits = [['name', 'MOT16-06'], ['imDir', 'img1'],
            # ['frameRate', '14'],
            # ['seqLength', '1194'], ['imWidth', '640'], ['imHeight', '480'],
            # ['imExt', '.jpg'], ['']]
            line_splits = [l.split('=') for l in f.read().splitlines()[1:]]
            info_dict = dict(
                s for s in line_splits if isinstance(s, list) and len(s) == 2)

        update_ms = 1000 / int(info_dict["frameRate"])
    else:
        update_ms = None
    # detections.shape = (10853, 138)
    feature_dim = detections.shape[1] - 10 if detections is not None else 0
    # --sequence_dir =./ MOT16 / test / MOT16 - 06
    # os.path.basename
    # 函数作用：返回path最后的文件名
    seq_info = {
        #sequence_name: MOT16-06
        "sequence_name": os.path.basename(sequence_dir),
        "image_filenames": image_filenames,
        "detections": detections,
        "groundtruth": groundtruth,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "feature_dim": feature_dim,
        "update_ms": update_ms
    }
    return seq_info


def create_detections(detection_mat, frame_idx, min_height=0):
    """
    从原始检测矩阵为给定的帧索引创建检测。
    Create detections for given frame index from the raw detection matrix.

    Parameters
    ----------
    detection_mat : ndarray
    检测矩阵。
    检测矩阵的前10列采用标准的MOTChallenge检测格式。
    在其余的列中存储与每个检测相关联的特征向量。
        Matrix of detections. The first 10 columns of the detection matrix are
        in the standard MOTChallenge detection format. In the remaining columns
        store the feature vector associated with each detection.
    frame_idx : int
        The frame index.
    min_height : Optional[int]
    最小检测边界框高度。小于此值的检测将被忽略。
        A minimum detection bounding box height. Detections that are smaller
        than this value are disregarded.

    Returns
    -------
    List[tracker.Detection]
        Returns detection responses at given frame index.

    """
    frame_indices = detection_mat[:, 0].astype(np.int)
    # mask是一个布尔型矩阵数组，True的位置是与frame_id匹配的位置
    mask = frame_indices == frame_idx

    detection_list = []
    for row in detection_mat[mask]:
        # 可知detection_mat的数据格式为
        # 0： idx
        # 1：？？都是-1？？？
        # 2:5：xywh bbox参数
        #6： 置信度
        #7- ：特征
        bbox, confidence, feature = row[2:6], row[6], row[10:]
        if bbox[3] < min_height:
            continue
        detection_list.append(Detection(bbox, confidence, feature))
    return detection_list


def run(sequence_dir, detection_file, output_file, min_confidence,
        nms_max_overlap, min_detection_height, max_cosine_distance,
        nn_budget, display):
    """Run multi-target tracker on a particular sequence.

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detections file.
    output_file : str
        Path to the tracking output file. This file will contain the tracking
        results on completion.
    min_confidence : float
        Detection confidence threshold. Disregard all detections that have
        a confidence lower than this value.
    nms_max_overlap: float
        Maximum detection overlap (non-maxima suppression threshold).
    min_detection_height : int
        Detection height threshold. Disregard all detections that have
        a height lower than this value.
    max_cosine_distance : float
        Gating threshold for cosine distance metric (object appearance).
    nn_budget : Optional[int]
        Maximum size of the appearance descriptor gallery. If None, no budget
        is enforced.
    display : bool
        If True, show visualization of intermediate tracking results.

    """
    seq_info = gather_sequence_info(sequence_dir, detection_file)
    # NearestNeighborDistanceMetric类
    # max_cosine_distance默认0.2，nn_budget默认None
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    results = []
    def frame_callback(vis, frame_idx):
        # 此vis是Visualization类对象
        print("Processing frame %05d" % frame_idx)

        # Load image and generate detections.
        # 取得idx对应的特征列表，列表中的元素以Detection对象形式存在

        detections = create_detections(
            seq_info["detections"], frame_idx, min_detection_height)
        is_last_frame = frame_idx+1 <= seq_info['max_frame_idx']
        if is_last_frame:
            next_frame_detections = create_detections(
            seq_info["detections"], frame_idx+1, min_detection_height)
        else:
            next_frame_detections = []
        # 通过置信度对取得的特征进行筛选
        # bbox, confidence, feature = row[2:6], row[6], row[10:]
        detections = [d for d in detections if d.confidence >= min_confidence]
        next_frame_detections = [d for d in next_frame_detections if d.confidence >= min_confidence]

        # Run non-maxima suppression.
        # 边界框集合列表
        boxes = np.array([d.tlwh for d in detections])
        next_frame_boxes = np.array([d.tlwh for d in next_frame_detections])
        # 置信度列表
        scores = np.array([d.confidence for d in detections])
        next_frame_scores = np.array([d.confidence for d in next_frame_detections])
        # 非极大抑制，为了将重叠度过高的候选框去掉
        indices = preprocessing.non_max_suppression(
            boxes, nms_max_overlap, scores)
        next_frame_indices = preprocessing.non_max_suppression(
            next_frame_boxes, nms_max_overlap, next_frame_scores)
        detections = [detections[i] for i in indices]
        next_frame_detections = [next_frame_detections[i] for i in next_frame_indices]

        if frame_idx == 449:
            print('kk')
        tracker.detection_record['{}'.format(frame_idx)] = {}
        for detection_idx in range(len(detections)):
            tracker.detection_record['{}'.format(frame_idx)] \
                ['{}'.format(detection_idx)] = []
        # Update tracker.
        tracker.start_new_tracks()
        undealed_detections = tracker.start_tracks(detections,frame_idx)
        # 对现有的所有轨迹进行预测处理，得到每个轨迹关联门内的检测索引
        tracker.predict()
        # 计算关联概率
        tracker.cal_jpda_prob(detections, frame_idx, threshold=0.0)
        # jpda_prob_matrix = tracker.cal_jpda_prob(detections,frame_idx,threshold=0.0)
        # tracker.update(jpda_prob_matrix,detections,undealed_detections,next_frame_detections,frame_idx)
        tracker.match(detections,undealed_detections,next_frame_detections,frame_idx)
        # 如果连续多帧都存在两个或多个轨迹重合在一个检测上，如（9，25）、（9，25）、（9，25）则只处理一次即可
        # tracker.consolidate_detection_records(frame_idx)
        interval = 10
        track_episode_length = 30
        if frame_idx == 400:
            print("s")
        if frame_idx % interval == 0:
            temp1_detection_dict = tracker.get_unnormal_points()
            dealed_track_reord = tracker.post_process_per_n_frames(temp1_detection_dict,frame_idx,interval,track_episode_length)
            tracker.update_tracks(dealed_track_reord)
        # Update visualization.
        # if display:
        #     image = cv2.imread(
        #         seq_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR)
        #     vis.set_image(image.copy())
        #     vis.draw_detections(detections)
        #     vis.draw_trackers(tracker.tracks)
        # # Store results.
        # for track in tracker.tracks:
        #     if not track.is_confirmed() or track.time_since_update > 1:
        #         continue
        #     bbox = track.to_tlwh()
        #     results.append([
        #         frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])

    def frame_callback_show(vis, frame_idx):
        # 此vis是Visualization类对象
        print("Showing frame %05d" % frame_idx)
        # Load image and generate detections.
        # 取得idx对应的特征列表，列表中的元素以Detection对象形式存在
        detections = create_detections(
            seq_info["detections"], frame_idx, min_detection_height)
        # 通过置信度对取得的特征进行筛选
        # bbox, confidence, feature = row[2:6], row[6], row[10:]
        detections = [d for d in detections if d.confidence >= min_confidence]

        # Run non-maxima suppression.
        # 边界框集合列表
        boxes = np.array([d.tlwh for d in detections])
        # 置信度列表
        scores = np.array([d.confidence for d in detections])
        # 非极大抑制，为了将重叠度过高的候选框去掉
        indices = preprocessing.non_max_suppression(
            boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        # Update visualization.
        if display:
            image = cv2.imread(
                seq_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR)
            vis.set_image(image.copy())
            vis.draw_detections(detections)
            vis.draw_trackers_by_tracks_records(tracker.tracks_record,frame_idx)
        # Store results.
        # for track in tracker.tracks:
        #     if not track.is_confirmed() or track.time_since_update > 1:
        #         continue
        #     bbox = track.to_tlwh()
        #     results.append([
        #         frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])
    # Run tracker.
    if display:
        # 创建Visualization类
        visualizer1 = visualization.Visualization(seq_info, update_ms=100)
        visualizer2 = visualization.Visualization(seq_info, update_ms=100)
    else:
        # 创建NoVisualization类
        visualizer = visualization.NoVisualization(seq_info)
    # visualizer.run(frame_callback)
    visualizer1.deal(frame_callback)
    tracker.remove_fake_tracks(5)
    visualizer2.run(frame_callback_show)
       # tracker.post_process()


    # Store results.
    # f = open(output_file, 'w')
    if not os.path.exists('tmp'):
        os.makedirs('tmp')
    with open(output_file,'w') as f:
        for row in results:
            print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
                row[0], row[1], row[2], row[3], row[4], row[5]),file=f)




def bool_string(input_string):
    if input_string not in {"True","False"}:
        raise ValueError("Please Enter a valid Ture/False choice")
    else:
        return (input_string == "True")

def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Deep SORT")
    # MOTChallenge序列目录路径
    '''
    --sequence_dir =./ MOT16 / test / MOT16 - 06 
    - -detection_file =./ resources / detections / MOT16_POI_test / MOT16 - 06.npy
     - -min_confidence = 0.3 
     - -nn_budget = 100 
     - -display = True
    
    '''
    #- -min_confidence = 0.3 - -nn_budget = 100 - -display = True
    parser.add_argument("--sequence_dir", help="Path to MOTChallenge sequence directory",
        default='./MOT16/test/MOT16-01/')
    # 自定义检测的路径
    parser.add_argument("--detection_file", help="Path to custom detections.",
        default='./resources/detections/MOT16_POI_test/MOT16-01.npy')
    # 跟踪输出文件的路径。该文件将包含完成时的跟踪结果。
    parser.add_argument("--output_file", help="Path to the tracking output file. This file will"
        " contain the tracking results on completion.",
        default="./tmp/hypotheses.txt")
    #检测置信阈值，忽略所有置信度低于此值的检测结果。
    parser.add_argument("--min_confidence", help="Detection confidence threshold. Disregard "
        "all detections that have a confidence lower than this value.",
        default=0.3, type=float)
    # 检测边界框的高度阈值。高度小于此值的检测被忽略
    parser.add_argument("--min_detection_height", help="Threshold on the detection bounding "
        "box height. Detections with height smaller than this value are "
        "disregarded", default=0, type=int)
    # 非极大抑制阈值：最大检测重叠
    parser.add_argument("--nms_max_overlap",  help="Non-maxima suppression threshold: Maximum "
        "detection overlap.", default=1.0, type=float)
    # 最大欧式距离
    parser.add_argument("--max_cosine_distance", help="Gating threshold for cosine distance "
        "metric (object appearance).", type=float, default=0.0)
    #外观描述符库的最大大小。如果为None，则不执行任何开支。
    parser.add_argument("--nn_budget", help="Maximum size of the appearance descriptors "
        "gallery. If None, no budget is enforced.", type=int, default=100)
    parser.add_argument("--display", help="Show intermediate tracking results",
        default=True, type=bool_string)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.sequence_dir, args.detection_file, args.output_file,
        args.min_confidence, args.nms_max_overlap, args.min_detection_height,
        args.max_cosine_distance, args.nn_budget, args.display)

    """
    python3 deep_sort_app.py \
    --sequence_dir=./MOT16/test/MOT16-08 \
    --detection_file=./resources/detections/MOT16_POI_test/MOT16-08.npy \
    --min_confidence=0.3 \
    --nn_budget=100 \
    --display=True
    """
