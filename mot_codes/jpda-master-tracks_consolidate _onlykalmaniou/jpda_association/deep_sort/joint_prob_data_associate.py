import numpy as np

from jpda_association.application_util.math_operation import MathOperation
class Joint_prob_data_associate:
    def __init__(self):
        # 需要当前的轨迹集合及其关联门内的量测索引，需要当前帧的量测
        self.math_operation = MathOperation()

    def update_tracks_and_detections(self,tracks,detections,threshold):
        self.tracks = tracks
        self.detections = detections
        self.threshold = threshold

    # 生成确认矩阵
    def generate_associate_matrix(self,dynamic_model_num):
        self.associate_matrix = np.zeros((len(self.detections),len(self.tracks)))
        for track_i,track in enumerate(self.tracks):
            for detection_ind in track.associated_detection_list:
                self.associate_matrix[detection_ind,track_i] = 1

    # 生成联合事件,极大连通图
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
                                traversed_col_row.append([track_id, t])
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
                if len(associate_track_id) != 0:
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


        # 为每一个联合事件（连通图）生成可行事件sita(i)的矩阵
    def generate_associate_event_matrixs(self, associate_events_tracks, associate_events_detections):
        associate_event_matrix = np.zeros((self.associate_matrix.shape[0], self.associate_matrix.shape[1] + 1))
        associate_event_matrix[:, 0] = 1
        # 生成关联矩阵
        for detection in associate_events_detections:
            for track in associate_events_tracks:
                if self.associate_matrix[detection][track] == 1:
                    associate_event_matrix[detection][track + 1] = 1

        associate_event_matrix_list = []
        associate_event_matrix_list_temp = []
        # 对关联矩阵生成关联事件
        for detection_id in range(associate_event_matrix.shape[0]):
            if len(associate_event_matrix_list) != 0:
                for i in range(len(associate_event_matrix_list)):
                    a_associate_matrix = associate_event_matrix_list[i].copy()
                    for track_id in range(associate_event_matrix.shape[1]):
                        if (associate_event_matrix[detection_id, track_id] == 1 and track_id == 0) or \
                                (associate_event_matrix[detection_id, track_id] == 1 and \
                                 sum(a_associate_matrix[:, track_id]) == 0):
                            a_associate_event = np.zeros_like(associate_event_matrix)
                            a_associate_event = a_associate_matrix.copy()
                            a_associate_event[detection_id, track_id] = 1
                            associate_event_matrix_list_temp.append(a_associate_event)
                associate_event_matrix_list = []
                while len(associate_event_matrix_list_temp) != 0:
                    associate_event_matrix_list.append(associate_event_matrix_list_temp.pop(0))
            else:
                for track_id in range(associate_event_matrix.shape[1]):
                    if associate_event_matrix[detection_id, track_id] == 1:
                        a_associate_event = np.zeros_like(associate_event_matrix)
                        a_associate_event[detection_id, track_id] = 1
                        associate_event_matrix_list.append(a_associate_event)
        return  associate_events_detections,associate_events_tracks,associate_event_matrix_list

    # 计算beta(jt)概率
    def cal_beta_of_cur_associate_event(self,associate_events_jpda_prob_matrix, associate_events_detections, associate_events_tracks,
                                        associate_event_matrix_list):
        normalized_constant_c = 0.0
        # 求归一化常数c
        associate_event_matrix_prob_list = []
        # 遍历所有可行事件
        for i in range(len(associate_event_matrix_list)):
            a_associate_event_matrix = associate_event_matrix_list[i]
            #先计算tao, sita, rou的值
            # 量测互联指示
            tao_array = [0 for _ in range(a_associate_event_matrix.shape[0])]
            # 目标检测指示
            sita_array = [0 for _ in range(a_associate_event_matrix.shape[1])]
            # 联合事件中假量测的数量
            rou = 0
            # 量测互联指示
            for j in range(a_associate_event_matrix.shape[0]):
                row = a_associate_event_matrix[j,:]
                tao_array[j] = sum(row[1:])
            # 目标检测指示
            for j in range(1,a_associate_event_matrix.shape[1]):
                sita_array[j] = sum(a_associate_event_matrix[:,j])
            for j in associate_events_detections:
                rou = rou + (1-tao_array[j])

            prob_sita_1 = self.math_operation.cal_factorial(rou)/self.math_operation.cal_factorial(len(associate_events_detections))
            # 非均匀分布时nunbda是虚假量测的空间密度，nunbda*V是门内虚假量测期望数量
            # 均匀分布时,prob_sita_2_Uf为常数
            # 这里用非均匀分布,nunbda*V设为1
            # 关联门体积，先设置为sqrt(30)
            gate_volume = np.sqrt(30)
            nunbda_V  = gate_volume
            prob_sita_2_Uf = np.exp(-nunbda_V)*((nunbda_V)**rou/self.math_operation.cal_factorial(rou))
            prob_sita_3 = 1
            for j in associate_events_tracks:
                prob_sita_3 = prob_sita_3*((self.tracks[j].detect_prob)**sita_array[j+1])*\
                              ((1-self.tracks[j].detect_prob)**(1 - sita_array[j+1]))

            hsh = np.array([[14.4375,0.0000,0.00000,0.0000],
                            [0.0000,14.4375,0.00000,0.0000],
                            [0.0000,0.0000,1.01001,0.0000],
                            [0.0000,0.0000,0.00000,8.3125]
                            ])
            Volume = ((np.linalg.det(2*np.pi*hsh))**(0.5))/(np.exp(-0.5*0.01))
            f_prob_sita_zk =Volume**(-1*rou)*0.1
            f_prob_normal_distribution = 1
            for detection_ind in associate_events_detections:
                for track_ind in associate_events_tracks:
                    if a_associate_event_matrix[detection_ind, track_ind+1] == 1:
                        index = self.tracks[track_ind].associated_detection_list.index(detection_ind)
                        model_index = int(self.tracks[track_ind].model_index_list[index])

                        # np.dot(model.get_kalman_filter_X0(), model.get_H_matrix().T)
                        hx = np.dot(self.tracks[track_ind].imm_controller.model_list[model_index].get_kalman_filter_X0(),\
                                    self.tracks[track_ind].imm_controller.model_list[model_index].get_H_matrix().T)
                        dv = hx-self.detections[detection_ind].to_xyah()
                        # covariance_t = np.linalg.multi_dot((
                        #                 model.get_H_matrix(), model.get_kalman_filter_P0(), model.get_H_matrix().T))+model.get_R_matrix()
                        s = np.linalg.multi_dot((
                            self.tracks[track_ind].imm_controller.model_list[model_index].get_H_matrix(), \
                            self.tracks[track_ind].imm_controller.model_list[model_index].get_kalman_filter_P0(), \
                            self.tracks[track_ind].imm_controller.model_list[model_index].get_H_matrix().T))+\
                            self.tracks[track_ind].imm_controller.model_list[model_index].get_R_matrix()
                        dv_s_dv = np.linalg.multi_dot((dv,np.linalg.inv(s),dv.T))
                        normal_distribution = (1/(np.linalg.det(2*np.pi*s))**(0.5))*np.exp(-0.5*dv_s_dv)
                        f_prob_normal_distribution = f_prob_normal_distribution*normal_distribution
                        f_prob_normal_distribution = f_prob_normal_distribution[0,0]*(1-0.1)/(len(associate_event_matrix_list)-1)*10
                        break
            joint_prob = prob_sita_1 * prob_sita_2_Uf * prob_sita_3 * f_prob_sita_zk * f_prob_normal_distribution
            associate_event_matrix_prob_list.append(joint_prob)

            for detection_ind in associate_events_detections:
                for track_ind in associate_events_tracks:
                    if a_associate_event_matrix[detection_ind, track_ind + 1] == 1:
                        associate_events_jpda_prob_matrix[detection_ind+1, track_ind] = \
                            associate_events_jpda_prob_matrix[detection_ind+1, track_ind]+joint_prob
            # prob(0->track)
            # 对于关联事件中，没有量测源于目标的概率，概率矩阵中加入0行代表没有量测来源于目标
            for track_ind_1 in associate_events_tracks:
                if sita_array[track_ind_1+1] == 0:
                    associate_events_jpda_prob_matrix[0,track_ind_1] = \
                        associate_events_jpda_prob_matrix[0,track_ind_1] +joint_prob
        #此时已经得到了当前联合事件中每个可行事件的概率列表associate_event_matrix_prob_list
        #且associate_events_jpda_prob_matrix矩阵存储了当前联合事件中每个可行事件中的对应的轨迹-检测对的关联概率
        normalized_constant_c = np.sum(associate_event_matrix_prob_list) if np.sum(associate_event_matrix_prob_list)>0.0 else 1
        return associate_events_jpda_prob_matrix/normalized_constant_c
























