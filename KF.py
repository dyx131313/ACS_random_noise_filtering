import numpy as np
import matplotlib.pyplot as plt
import csv
from data import train_dataset
from plot_fun.noised_prediction_test_plot import plot_prediction  # �����ͼ����

# ��ȡ���ݼ�
class Dataset:
    def __init__(self, csv_file):
        self.points = self.load_from_csv(csv_file)

    def load_from_csv(self, csv_file):
        points = []
        with open(csv_file, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # ������ͷ
            for row in reader:
                if row:  # ������Ƿ�Ϊ��
                    points.append([float(row[0]), float(row[1])])
        return np.array(points)

# ����������
def add_noise(points, noise_level=0.1):
    noise = np.random.normal(0, noise_level, points.shape)
    return points + noise

# Kalman Filter implementation
class KalmanFilter:
    def __init__(self, label = "default", dt = 1, process_noise = 0.01, measurement_noise = 0.2):
        self.dt = dt
        self.state_dim = 2  # ֻ���� y ���ϵ�λ�ú��ٶ�
        self.measurement_dim = 1  # ֻ���� y ���ϵ�λ��
        self.A = np.array([[1, dt], [0, 1]])  # ״̬ת�ƾ���
        self.H = np.array([[1, 0]])  # ��������
        self.Q = process_noise * np.eye(self.state_dim)  # ��������Э�������
        self.R = measurement_noise * np.eye(self.measurement_dim)  # ��������Э�������
        self.x_hat = np.zeros(self.state_dim)  # ��ʼ״̬����
        self.P = np.eye(self.state_dim)  # ��ʼ����Э�������
        self.gen_noise = np.random.normal(0, 0.1, 1000)
        self.label = label

    def predict(self):
        self.x_hat = self.A @ self.x_hat
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, measurement):
        y = measurement - self.H @ self.x_hat
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x_hat = self.x_hat + K @ y
        self.P = (np.eye(self.state_dim) - K @ self.H) @ self.P

    def run(self, measurements):
        estimated_trajectory = []
        for measurement in measurements:
            self.predict()
            self.update(measurement)
            estimated_trajectory.append(self.x_hat[0])  # ֻ���� y ���ϵ�λ��Ԥ��ֵ
        return np.array(estimated_trajectory)
    
    def noised(self, points, is_noised=1):
        ret_points = points.copy()
        if is_noised == 0:
            return None, ret_points
        noise = np.random.choice(self.gen_noise, size=ret_points[:, 1].shape)
        ret_points[:, 1] += noise
        return noise, ret_points

    def process_dataset(self, dataset, noise_level=0.1, output_dir='KF_data'):
        points = dataset.points
        _, points_noised = self.noised(points)
        measurements = points_noised[:, 1]  # ֻ���� y ���ϵĲ���ֵ
        estimated_trajectory = self.run(measurements)

        # �� x ���ԭʼ������ y ���Ԥ��ֵ�������
        estimated_trajectory_with_x = np.column_stack((points[:, 0], estimated_trajectory))
        
        # ֻȡ���һ��Ԥ���
        last_estimated_point = estimated_trajectory_with_x[-1]

        # ��������CSV�ļ�
        with open(f'{output_dir}/points/{self.label}_{dataset.label}_points.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Original Points', 'Predicted Point', 'Loss'])
            for point, est_point in zip(points, estimated_trajectory_with_x):
                writer.writerow([point, est_point, np.linalg.norm(point[1] - est_point[1])])

        # ֻȡ���һ��Ԥ������ʧ
        cur_loss = np.linalg.norm(points[-1, 1] - last_estimated_point[1])
        
        # ��ͼ
        plot_prediction(points, points_noised, last_estimated_point, label=dataset.label, output_path=f'{output_dir}/plot/{self.label}_{dataset.label}_pred.png', loss = cur_loss)
    def train(self):
        for data in train_dataset:
            self.process_dataset(data)
# ����Kalman Filter
if __name__ == "__main__":
    X_pred = KalmanFilter(label = "X")
    X_pred.train()
    
    Y_pred = KalmanFilter(label = "Y")
    Y_pred.train()
