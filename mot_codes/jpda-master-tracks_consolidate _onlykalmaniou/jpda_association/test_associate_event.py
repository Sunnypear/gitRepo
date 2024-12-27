import unittest

import numpy as np


class TestAssociateEvent:
    def __init__(self,associate_matrix, tracks, detections, threshold):
        # associate_matrix: 列号是轨迹号，行号是检测号
        self.associate_matrix = associate_matrix
        self.tracks = tracks
        self.detections = detections
        self.threshold = threshold


    def generate_associate_events(self):
        traversed_col_row = []
        traversed_detection_id = []

        associate_track_id = []
        traversed_track_id = []
        traversed_col_row.append([0, 0])
        associate_events = []
        col = 0
        keep = True
        while keep:
            if len(traversed_col_row) != 0:
                # col = -
                a_associate_event_track_id = []
                a_associate_event_detection_id = []
                while len(traversed_col_row) != 0:
                    [track_id, detection_id] = traversed_col_row.pop(0)
                    if track_id not in traversed_track_id:
                        for t in range(self.associate_matrix.shape[0]):
                            if self.associate_matrix[t, track_id] == 1:
                                traversed_col_row.append([track_id,t])
                        traversed_track_id.append(track_id)
                        associate_track_id.append(track_id)
                    else:
                        if detection_id not in traversed_detection_id:
                            for s in range(self.associate_matrix.shape[1]):
                                if self.associate_matrix[detection_id, s] == 1:
                                    traversed_col_row.append([s, detection_id])
                            traversed_detection_id.append(detection_id)
                for track_idd in associate_track_id:
                    a_associate_event_track_id.append(track_idd)
                for detection_idd in traversed_detection_id:
                    a_associate_event_detection_id.append(detection_idd)
                associate_events.append(
                    {'tracks': a_associate_event_track_id, 'detections': a_associate_event_detection_id})
                associate_track_id = []
                traversed_detection_id = []
            elif col not in traversed_track_id:
                for j in range(self.associate_matrix.shape[0]):
                    if self.associate_matrix[j, col] == 1:
                        traversed_col_row.append([col, j])
                traversed_track_id.append(col)
                associate_track_id.append(col)
            else:
                if len(associate_track_id) !=0:
                    associate_events.append(
                        {'tracks': associate_track_id, 'detections': traversed_detection_id})
                    associate_track_id = []
                    traversed_detection_id = []
                # 找下一个尚未关联的track
                flag = False
                for a in range(self.associate_matrix.shape[1]):
                    if a not in traversed_track_id:
                        col = a
                        flag = True
                # 说明所有track均被关联，可以结束
                if not flag:
                    keep = False
        return associate_events

    def generate_associate_event_matrixs(self,associate_events_tracks, associate_events_detections):
        associate_event_matrix = np.zeros((self.associate_matrix.shape[0],self.associate_matrix.shape[1]+1))
        associate_event_matrix[:,0] = 1
        # 生成关联矩阵
        for detection in associate_events_detections:
            for track in associate_events_tracks:
                if self.associate_matrix[detection][track] == 1:
                    associate_event_matrix[detection][track+1] = 1

        associate_event_matrix_list = []
        associate_event_matrix_list_temp = []
        # 对关联矩阵生成关联事件
        for detection_id in range(associate_event_matrix.shape[0]):
            if len(associate_event_matrix_list) !=0:
                for i in range(len(associate_event_matrix_list)):
                    a_associate_matrix = associate_event_matrix_list[i].copy()
                    for track_id in range(associate_event_matrix.shape[1]):
                        if (associate_event_matrix[detection_id, track_id] == 1 and track_id ==0) or\
                        (associate_event_matrix[detection_id, track_id] == 1 and \
                         sum(a_associate_matrix[:,track_id]) == 0):
                            a_associate_event = np.zeros_like(associate_event_matrix)
                            a_associate_event = a_associate_matrix.copy()
                            a_associate_event[detection_id,track_id] = 1
                            associate_event_matrix_list_temp.append(a_associate_event)
                associate_event_matrix_list = []
                while len(associate_event_matrix_list_temp) !=0:
                    associate_event_matrix_list.append(associate_event_matrix_list_temp.pop(0))
            else:
                for track_id in range(associate_event_matrix.shape[1]):
                    if associate_event_matrix[detection_id, track_id] == 1:
                        a_associate_event = np.zeros_like(associate_event_matrix)
                        a_associate_event[detection_id, track_id] = 1
                        associate_event_matrix_list.append(a_associate_event)
        # return associate_events_detections, associate_events_tracks, associate_event_matrix_list
        print(associate_event_matrix)





def __main__():
    associate_matrix = np.array([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0],\
                                 [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],\
                                 [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],\
                                 [0, 0, 1, 0, 1, 0, 1, 0, 0, 0],\
                                 [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],\
                                 [1, 0, 0, 0, 0, 0, 1, 0, 0, 0],\
                                 [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],\
                                 [1, 0, 0, 0, 0, 0, 0, 0, 1, 0],\
                                 [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],\
                                 [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]])
    # associate_matrix = np.zeros((36,36))
    # associate_matrix[21,0] = 1
    # associate_matrix[21,1] = 1
    # associate_matrix[25,2] = 1
    # associate_matrix[23,3] = 1
    # associate_matrix[27,4] = 1
    # associate_matrix[22,5] = 1
    # associate_matrix[28,6] = 1
    # associate_matrix[20, 7] = 1
    # associate_matrix[19, 8] = 1
    # associate_matrix[26, 9] = 1
    # associate_matrix[29, 9] = 1
    # associate_matrix[24, 10] = 1
    # associate_matrix[35, 11] = 1
    # associate_matrix[24, 12] = 1
    # associate_matrix[24, 13] = 1
    # associate_matrix[33, 14] = 1
    # associate_matrix[24, 16] = 1
    # associate_matrix[28, 6] = 1
    tracks = []
    detections = []
    threshold =0.0
    print(associate_matrix.shape)
    test_associate_event = TestAssociateEvent(associate_matrix, tracks, detections, threshold)
    associate_event=test_associate_event.generate_associate_events()
    for a_associate_event in associate_event:
        test_associate_event.generate_associate_event_matrixs(a_associate_event['tracks'],a_associate_event['detections'])
    print(associate_event)

__main__()