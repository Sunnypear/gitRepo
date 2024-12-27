import numpy as np
class Track:
    def __init__(self,X0,dt,F_matrix,L_matrix,H_matrix,M_matrix):
        # (1,4)
        self.X0 = X0
        self.dt = dt
        self.track = []
        # (4,4)
        self.F_matrix = F_matrix
        # (4,2)
        self.L_matrix = L_matrix
        # (2，4)
        self.H_matrix = H_matrix
        # (2,2)
        self.M_matrix = M_matrix

    def generate_track(self,time,w,v):
        for t in range(time):
            # （4，1）
            # kalman_Hx0 = np.dot(self.F_matrix,self.X0.T)+self.L_matrix@w.T
            kalman_Hx0 = np.dot(self.F_matrix, self.X0.T)
            # （2，1）
            # measurement = np.dot(self.H_matrix,kalman_Hx0) + self.M_matrix@v.T
            measurement = np.dot(self.H_matrix, kalman_Hx0)
            self.X0 = kalman_Hx0.T
            self.track.append((measurement.T)[0])
        return self.track
    def get_track(self):
        return self.track





