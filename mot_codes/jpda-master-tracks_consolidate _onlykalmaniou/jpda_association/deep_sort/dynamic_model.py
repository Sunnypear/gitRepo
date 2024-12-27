

class Dynamic_model:
    def __init__(self, prob_mui, mean_xk0, convariance_xk0,F_matrix,H_matrix,Q_matrix,R_matrix,model_max_velocity=[8,8,0,0]):
        self.prob_mui = prob_mui
        self.F_matrix = F_matrix
        self.H_matrix = H_matrix
        self.Q_matrix = Q_matrix
        self.R_matrix = R_matrix
        self.model_max_velocity = model_max_velocity
        self.possible_measure_index = []
        self.possible_measure_distance = []
        self.kalman_filter_X0 = []
        self.kalman_filter_P0 = []
        self.update_mean_X = mean_xk0
        self.update_covariance_P = convariance_xk0
        self.model_prob_estimate = 0.0




    def get_prob_mui(self):
        return self.prob_mui
    def get_mean_xk0(self):
        return self.update_mean_X
    def get_convariance_xk0(self):
        return self.update_covariance_P
    def get_F_matrix(self):
        return self.F_matrix
    def get_H_matrix(self):
        return self.H_matrix
    def get_Q_matrix(self):
        return self.Q_matrix
    def get_R_matrix(self):
        return self.R_matrix
    def get_model_max_velocity(self):
        return self.model_max_velocity
    def get_possible_measure_index(self):
        return self.possible_measure_index
    def get_possible_measure_distance(self):
        return self.possible_measure_distance
    def get_kalman_filter_X0(self):
        return self.kalman_filter_X0
    def get_kalman_filter_P0(self):
        return self.kalman_filter_P0
    def get_update_mean_X(self):
        return self.update_mean_X
    def get_update_covariance_P(self):
        return self.update_covariance_P
    def get_model_prob_estimate(self):
        return self.model_prob_estimate
    def set_update_mean_X(self, update_mean_X):
        self.update_mean_X = update_mean_X
    def set_update_covariance_P(self, update_covariance_P):
        self.update_covariance_P = update_covariance_P
    def set_model_prob_estimate(self, model_prob_estimate):
        self.model_prob_estimate = model_prob_estimate
    def set_prob_mui(self,prob_mui):
        self.prob_mui = prob_mui

    def set_F_matrix(self,F_matrix):
        self.F_matrix = F_matrix
    def set_H_matrix(self,H_matrix):
        self.H_matrix = H_matrix
    def set_Q_matrix(self,Q_matrix):
        self.Q_matrix = Q_matrix
    def set_R_matrix(self,R_matrix):
        self.R_matrix = R_matrix
    def set_model_max_velocity(self,model_max_velocity):
        self.model_max_velocity = model_max_velocity
    def set_possible_measure_index(self,possible_measure_index):
        self.possible_measure_index = possible_measure_index
    def set_possible_measure_distance(self,possible_measure_distance):
        self.possible_measure_distance = possible_measure_distance
    def set_kalman_filter_X0(self,kalman_filter_X0):
        self.kalman_filter_X0 = kalman_filter_X0
    def set_kalman_filter_P0(self,kalman_filter_P0):
        self.kalman_filter_P0 = kalman_filter_P0




