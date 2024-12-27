import numpy as np
class MathOperation:

    def cal_distance_bet_points(self,point1,point2):
        return np.sqrt((point2[0]-point1[0])**2+(point2[1]-point2[1])**2)


    def update_detection_record(self,detection_record,cur_track,origin_track,):
        detection_dict_sorted_index = sorted(detection_record.keys(), key=lambda k: int(k))
        for update_frame_ind in detection_dict_sorted_index:
            for a_detection in detection_record[update_frame_ind].keys():
                if origin_track in detection_record[update_frame_ind][a_detection]:
                    flag = 0
                    if cur_track in detection_record[update_frame_ind][a_detection]:
                        flag = 1
                    ind = detection_record[update_frame_ind][a_detection].index(origin_track)
                    detection_record[update_frame_ind][a_detection][ind] = cur_track
                    if flag == 1:
                        lab = 0
                        for tr in detection_record[update_frame_ind][a_detection]:
                            if tr == cur_track:
                                lab += 1
                        if len(detection_record[update_frame_ind][a_detection]) == lab:
                            detection_record[update_frame_ind][a_detection] = [cur_track]
        return  detection_record

    def is_in_posi1_and_posi2(self,cur_posi,posi1,posi2):
        if posi1 < posi2:
            if cur_posi >= posi1 and cur_posi <= posi2:
                return True
        elif posi1 > posi2:
            if cur_posi <= posi1 and cur_posi >= posi2:
                return True
        elif posi1 == posi2:
            if np.abs(cur_posi - posi1) <10:
                return True

    def get_dxdy(self,axis):
        # axis是一组按帧顺序排列的检测中心坐标值
        # 此函数是要通过这组坐标值求轨迹的增量
        max_dx = 0.
        max_dy = 0.
        for i in range(1,len(axis)):
            if abs(axis[i][0]- axis[i-1][0])>abs(max_dx):
                max_dx = axis[i][0]- axis[i-1][0]
            if abs(axis[i][1]- axis[i-1][1])>abs(max_dy):
                max_dy = axis[i][1]- axis[i-1][1]
        return max_dx,max_dy
    def cal_factorial(self,n):
        n = int(n)
        result = 1
        for i in range(n,0,-1):
            result = result * i
        return result
    def get_track_direction_sita(self,next_frame_detections_bcxywh,cur_detection_bcxywh):
        if next_frame_detections_bcxywh[1]==cur_detection_bcxywh[1]:
            if next_frame_detections_bcxywh[0]==cur_detection_bcxywh[0]:
                return 0*np.pi
            elif next_frame_detections_bcxywh[0]>cur_detection_bcxywh[0]:
                return np.pi/2
            else:
                return (3/2)*np.pi
        elif next_frame_detections_bcxywh[0]==cur_detection_bcxywh[0]:
            if next_frame_detections_bcxywh[1]==cur_detection_bcxywh[1]:
                return 0*np.pi
            elif next_frame_detections_bcxywh[1]>cur_detection_bcxywh[1]:
                return 0*np.pi
            else:
                return np.pi
        else:
            dy = np.abs(next_frame_detections_bcxywh[1]-cur_detection_bcxywh[1])
            dx = np.abs(next_frame_detections_bcxywh[0]-cur_detection_bcxywh[0])
            if next_frame_detections_bcxywh[1]>cur_detection_bcxywh[1] and \
                next_frame_detections_bcxywh[0]>cur_detection_bcxywh[0]:
                # 右下前进
                sitaa = np.arctan(dx/dy)
                t = np.sin(sitaa)
                return np.arctan(dx/dy)
            elif next_frame_detections_bcxywh[1]>cur_detection_bcxywh[1] and \
                next_frame_detections_bcxywh[0]<cur_detection_bcxywh[0]:
                # 左下前进
                sitaa = np.arctan(dy / dx)
                t = np.sin(sitaa)
                return (3/2)*np.pi+np.arctan(dy/dx)
            elif next_frame_detections_bcxywh[1]<cur_detection_bcxywh[1] and \
                next_frame_detections_bcxywh[0]>cur_detection_bcxywh[0]:
                # 右上前进
                sitaa = np.arctan(dy / dx)
                t = np.sin(sitaa)
                return (1/2)*np.pi+np.arctan(dy/dx)
            elif next_frame_detections_bcxywh[1]<cur_detection_bcxywh[1] and \
                next_frame_detections_bcxywh[0]<cur_detection_bcxywh[0]:
                # 右下前进
                sitaa = np.arctan(dx / dy)
                t = np.sin(sitaa)
                m = np.sin(np.pi/4)
                return np.pi+np.arctan(dx/dy)

    def get_min_row_col(self,matched_angles_temp):
        min_valr = [min(id) for id in matched_angles_temp]
        min_val = min(min_valr)
        for row in range(len(matched_angles_temp)):
            for col in range(len(matched_angles_temp[row])):
                if matched_angles_temp[row][col] == min_val:
                    return row, col


