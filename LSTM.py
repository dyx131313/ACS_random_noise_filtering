import torch
import random
import csv
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from data import train_dataset
from plot_fun.noised_prediction_test_plot import plot_prediction  # 导入绘图函数

class Trajectory_Prediction_Model:
    def __init__(self, label = "default", input_size=4, hidden_size=20, output_size=2, num_layers=3, dropout=0.3, lr=0.001, num_epochs=2000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.num_epochs = num_epochs

        self.model = self.LSTM(input_size, hidden_size, output_size, num_layers, dropout).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.gen_noise = np.random.normal(0, 0.1, 1000)

        self.label = label

    class LSTM(nn.Module):
        def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.5):
            super(Trajectory_Prediction_Model.LSTM, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
            self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
            self.fc2 = nn.Linear(hidden_size // 2, output_size)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            lstm_out, _ = self.lstm(x, (h0, c0))
            out = self.fc1(lstm_out[:, -1, :])  # 只取最后一个时间步的输出
            out = self.relu(out)
            out = self.dropout(out)
            out = self.fc2(out)
            return out

    def noised(self, points, is_noised=1):
        ret_points = points.copy()
        if is_noised == 0:
            return None, ret_points
        noise = np.random.choice(self.gen_noise, size=ret_points[:, 1].shape)
        ret_points[:, 1] += noise
        return noise, ret_points

    def shifted(self, points, is_shift=1):
        ret_points = points.copy()
        shift_x = 0.0
        if is_shift == 0:
            return shift_x, ret_points
        shift_x = ret_points[:, 1][-1]
        ret_points[:, 1] -= shift_x
        return shift_x, ret_points

    def normalized(self, points, is_normalized=1):
        ret_points = points.copy()
        norm = 1.0
        if is_normalized == 0:
            return norm, ret_points
        norm = np.linalg.norm(ret_points[:, 1])
        ret_points[:, 1] /= norm
        return norm, ret_points

    def form_y(self, points_diff, points_shift, points_norm):
        ret_points = points_diff.copy()[-1]
        ret_points -= points_shift
        ret_points /= points_norm
        return ret_points

    def train(self):
        for epoch in range(self.num_epochs):
            for data in train_dataset:
                points = data.points
                points_diff = np.diff(points, axis=0)
                _, points_noised = self.noised(points)
                points_shift_x, points_shifted = self.shifted(points_noised)
                points_shift_x = np.array([points_shift_x])
                points_norm, points_normalized = self.normalized(points_shifted)
                points_norm = np.array([points_norm])
                points_diff_ = np.diff(points_normalized, axis=0)

                points_shift_x_repeated = np.repeat(points_shift_x, points_diff_.shape[0], axis=0)
                points_norm_repeated = np.repeat(points_norm, points_diff_.shape[0], axis=0)
                points_shift_x_repeated = np.expand_dims(points_shift_x_repeated, axis=1)
                points_norm_repeated = np.expand_dims(points_norm_repeated, axis=1)

                combined_input = np.concatenate((points_diff_, points_shift_x_repeated, points_norm_repeated), axis=1)

                X = torch.tensor(combined_input[:-1], dtype=torch.float32).unsqueeze(0).to(self.device)  # 使用前19个点
                y = torch.tensor(points_diff[-1], dtype=torch.float32).unsqueeze(0).to(self.device)  # 预测第20个点
                self.optimizer.zero_grad()
                output = self.model(X)
                loss = self.criterion(output, y)
                loss.backward()
                self.optimizer.step()
                    
            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {loss.item():.4f}')


    def predict(self):
        self.model.eval()
        with torch.no_grad():
            for data in train_dataset:
                points = data.points
                points_diff = np.diff(points, axis=0)
                _, points_noised = self.noised(points)
                points_shift_x, points_shifted = self.shifted(points_noised)
                points_shift_x = np.array([points_shift_x])
                points_norm, points_normalized = self.normalized(points_shifted)
                points_norm = np.array([points_norm])
                points_diff_ = np.diff(points_normalized, axis=0)

                points_shift_x_repeated = np.repeat(points_shift_x, points_diff_.shape[0], axis=0)
                points_norm_repeated = np.repeat(points_norm, points_diff_.shape[0], axis=0)
                points_shift_x_repeated = np.expand_dims(points_shift_x_repeated, axis=1)
                points_norm_repeated = np.expand_dims(points_norm_repeated, axis=1)

                combined_input = np.concatenate((points_diff_, points_shift_x_repeated, points_norm_repeated), axis=1)
                X = torch.tensor(combined_input[:-1], dtype=torch.float32).unsqueeze(0).to(self.device)  # 使用前19个点
                y = torch.tensor(points_diff[-1], dtype=torch.float32).unsqueeze(0).to(self.device)  # 预测第20个点
                output = self.model(X)
                Loss = self.criterion(output, y)
                future_points_diff = output.squeeze().cpu().numpy()
                predicted_point = points[-2] + future_points_diff  # 预测第20个点

                # Save points, predicted_point, and loss to CSV
                with open(f'LSTM_data/points/{self.label}_{data.label}_points.csv', mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['Original Points', 'Predicted Point', 'Loss'])
                    for point in points:
                        writer.writerow([point, '', ''])
                    writer.writerow(['', predicted_point, Loss.item()])

                plot_prediction(points, points_noised, predicted_point, self.label, output_path=f'LSTM_data/plot/{self.label}_{data.label}_pred.png', loss=Loss.item())

# 使用示例
if __name__ == "__main__":
    X_prediction = Trajectory_Prediction_Model(label = "X")
    X_prediction.train()
    X_prediction.predict()
    Y_prediction = Trajectory_Prediction_Model(label = "Y")
    Y_prediction.train()
    Y_prediction.predict()
