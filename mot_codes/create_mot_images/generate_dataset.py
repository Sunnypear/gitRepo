# vim: expandtab:ts=4:sw=4
from __future__ import division, print_function, absolute_import
import argparse
import os
import random

import cv2
import numpy as np
from elements import track
from elements import detection
from elements import frame
from scipy.stats import multivariate_normal
from elements import visualization
from elements import image_viewer


def _generate_noise(covar):
    mean = np.zeros(covar.shape[0])
    return np.random.multivariate_normal(mean, covar)


def run(sequence_dir, detection_file, width,height,display,write):
    """Run multi-target tracker on a particular sequence.
    """
    ndim = 2
    dt = 1
    F_matrix = np.eye(2 * ndim, 2 * ndim)
    """
    [[1,0,0,0],
     [0,1,0,0],
     [0,0,1,0],
     [0,0,0,1]
    ]
    x0 :[x,dx,y,dy].T
    """
    for i in range(ndim):
        F_matrix[2 * i, 2 * i + 1] = dt
    """
    [[1,1,0,0],
     [0,1,0,0],
     [0,0,1,1],
     [0,0,0,1]
    ]
    """
    L = np.array([[0.5 * (dt ** 2), 0],
                       [dt, 0],
                       [0, 0.5 * (dt ** 2)],
                       [0, dt]])
    sigma_w = [1.5, 1.5]    # standard deviation of system noises.
    sigma_v =[1.5, 1.5]  # standard deviation of observation noises.
    M = np.eye(2)
    covar_ws = np.diag(np.array(sigma_w) ** 2)
    covar_vs = np.diag(np.array(sigma_v) ** 2)
    # Fx0 = [x+dx,dx,y+dy,dy]
    H_matrix = np.zeros((ndim, 2 * ndim))
    for i in range(ndim):
        H_matrix[i, 2 * i] = dt
    # [[1, 0, 0, 0],
    #  [0, 0, 1, 0]]
    Frames = []
    X0 = np.array([[12.,5.,20.,7.]])
    covariance = np.diag(np.array([5.,7.])**2)
    w = np.reshape(_generate_noise(covar_ws), (1, ndim))
    v = np.reshape(_generate_noise(covar_vs), (1, ndim))
    time = 200
    track1 = track.Track(X0,dt,F_matrix,L,H_matrix,M)
    track_list = track1.generate_track(time,w,v)
    track_list = [a for a in track_list if a[0]<width and a[1]<height]
    for t in range(len(track_list)):
        #随机虚假检测噪声
        img = np.zeros((height,width, 3), np.uint8)
        img.fill(255)
        a_frame = frame.Frame(t+1, img)
        det1 = detection.Detection([track_list[t][0],track_list[t][1],8,8],0.8)
        a_frame.add_detection(det1)
        is_noised = random.randint(0,1)
        if is_noised == 1:
            noised_num = random.randint(1,2)
            for i in range(noised_num):
                nosied = np.random.multivariate_normal(np.array(track_list[t]), covariance)
                confidence =multivariate_normal.pdf(track_list[t], track_list[t], covariance)
                det_temp = detection.Detection([nosied[0],nosied[1],8.,8.],0.6)
                a_frame.add_detection(det_temp)
        Frames.append(a_frame)

    label = []
    # img_path = './silmulation_imgs/sequence1/label.npy'
    for frame_index in range(len(Frames)):
        t_frame = Frames[frame_index]
        for adetection in t_frame.detections:
            label.append([t_frame.frame_id,*adetection.tlwh,adetection.confidence])

    np.save(detection_file, label)

    # temp = np.load(img_path)
    print("s")
    def frame_callback_show(vis, frame_idx):
        # 此vis是Visualization类对象
        print("Showing frame %05d" % frame_idx)
        # Load image and generate detections.
        # 取得idx对应的特征列表，列表中的元素以Detection对象形式存在
        if display:
            image = Frames[frame_idx].img
            # image = cv2.imread(
            #     Frames[frame_idx].img1, cv2.IMREAD_COLOR)
            vis.set_image(image.copy())
            vis.draw_detections(Frames[frame_idx].detections)
        if write:
            vis.write_to_img(sequence_dir,"0000"+"{}".format(frame_idx)+".jpg")
            # vis.draw_trackers_by_tracks_records(tracker.tracks_record,frame_idx)

    if display:
        # 创建Visualization类
        visualizer1 = visualization.Visualization(Frames,'modeling', update_ms=100)
        visualizer1.run(frame_callback_show)










def bool_string(input_string):
    if input_string not in {"True", "False"}:
        raise ValueError("Please Enter a valid Ture/False choice")
    else:
        return (input_string == "True")


def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="video information")
    # MOTChallenge序列目录路径
    parser.add_argument("--sequence_dir", help="Path to save",
                        default=r'./silmulation_imgs/sequence1/img1')
    # 自定义检测的路径
    parser.add_argument("--detection_file", help="Path to save detections.",
                        default=r'./silmulation_imgs/sequence1/label.npy')
    # # 自定义检测的路径
    parser.add_argument("--width", help="width of frame",default=1280,type=float)
    # 检测置信阈值，忽略所有置信度低于此值的检测结果。
    parser.add_argument("--height", help="height qf frame",default=1024,type=float)
    # # 检测边界框的高度阈值。高度小于此值的检测被忽略
    # parser.add_argument("--min_detection_height", help="Threshold on the detection bounding "
    #                                                    "box height. Detections with height smaller than this value are "
    #                                                    "disregarded", default=0, type=int)
    # # 非极大抑制阈值：最大检测重叠
    # parser.add_argument("--nms_max_overlap", help="Non-maxima suppression threshold: Maximum "
    #                                               "detection overlap.", default=1.0, type=float)
    # # 最大欧式距离
    # parser.add_argument("--max_cosine_distance", help="Gating threshold for cosine distance "
    #                                                   "metric (object appearance).", type=float, default=0.0)
    # # 外观描述符库的最大大小。如果为None，则不执行任何开支。
    # parser.add_argument("--nn_budget", help="Maximum size of the appearance descriptors "
    #                                         "gallery. If None, no budget is enforced.", type=int, default=100)
    parser.add_argument("--display", help="Show intermediate tracking results",
                        default=True, type=bool_string)
    parser.add_argument("--write", help="write intermediate tracking results",
                        default=True, type=bool_string)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.sequence_dir, args.detection_file, args.width,args.height,args.display,args.write)


