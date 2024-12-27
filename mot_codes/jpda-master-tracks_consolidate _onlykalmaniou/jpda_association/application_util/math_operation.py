import numpy as np
class MathOperation:
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


