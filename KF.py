import numpy as np
import matplotlib.pyplot as plt
import csv
from data import train_dataset
from plot_fun.noised_prediction_test_plot import plot_prediction  # 导入绘图函数

# 读取数据集
class Dataset:
    def __init__(self, csv_file):
        self.points = self.load_from_csv(csv_file)

    def load_from_csv(self, csv_file):
        points = []
        with open(csv_file, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # 跳过表头
            for row in reader:
                if row:  # 检查行是否为空
                    points.append([float(row[0]), float(row[1])])
        return np.array(points)

# 噪音处理函数
def add_noise(points, noise_level=0.1):
    noise = np.random.normal(0, noise_level, points.shape)
    return points + noise

# Kalman Filter implementation
class KalmanFilter:
    def __init__(self, label = "default", dt = 1, process_noise = 0.01, measurement_noise = 0.2):
        self.dt = dt
        self.state_dim = 2  # 只考虑 y 轴上的位置和速度
        self.measurement_dim = 1  # 只测量 y 轴上的位置
        self.A = np.array([[1, dt], [0, 1]])  # 状态转移矩阵
        self.H = np.array([[1, 0]])  # 测量矩阵
        self.Q = process_noise * np.eye(self.state_dim)  # 过程噪声协方差矩阵
        self.R = measurement_noise * np.eye(self.measurement_dim)  # 测量噪声协方差矩阵
        self.x_hat = np.zeros(self.state_dim)  # 初始状态估计
        self.P = np.eye(self.state_dim)  # 初始估计协方差矩阵
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
            estimated_trajectory.append(self.x_hat[0])  # 只保存 y 轴上的位置预测值
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
        measurements = points_noised[:, 1]  # 只考虑 y 轴上的测量值
        estimated_trajectory = self.run(measurements)

        # 将 x 轴的原始坐标与 y 轴的预测值结合起来
        estimated_trajectory_with_x = np.column_stack((points[:, 0], estimated_trajectory))
        
        # 只取最后一个预测点
        last_estimated_point = estimated_trajectory_with_x[-1]

        # 保存结果到CSV文件
        with open(f'{output_dir}/points/{self.label}_{dataset.label}_points.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Original Points', 'Predicted Point', 'Loss'])
            for point, est_point in zip(points, estimated_trajectory_with_x):
                writer.writerow([point, est_point, np.linalg.norm(point[1] - est_point[1])])

        # 只取最后一个预测点的损失
        cur_loss = np.linalg.norm(points[-1, 1] - last_estimated_point[1])
        
        # 绘图
        plot_prediction(points, points_noised, last_estimated_point, label=dataset.label, output_path=f'{output_dir}/plot/{self.label}_{dataset.label}_pred.png', loss = cur_loss)
    def train(self):
        for data in train_dataset:
            self.process_dataset(data)
# 运行Kalman Filter
if __name__ == "__main__":
    X_pred = KalmanFilter(label = "X")
    X_pred.train()
    
    Y_pred = KalmanFilter(label = "Y")
    Y_pred.train()
