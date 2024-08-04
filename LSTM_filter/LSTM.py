import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from LSTM_data import train_dataset
from plot_fun.noised_prediction_test_plot import plot_prediction  # 导入绘图函数


#noise geneartion
mean = 0
std_dev = 0.1
noise_size = 1000
gen_noise = np.random.normal(mean, std_dev, noise_size)

def noised(points, is_noised = 1):
    ret_points = points.copy()
    if(is_noised == 0):
      return None,ret_points
    noise = np.random.choice(gen_noise, size=ret_points[:,1].shape)
    ret_points[:,1] = ret_points[:,1] + noise
    return noise, ret_points

#data preprocessing
def shifted(points, is_shift = 1):
    ret_points = points.copy()
    shitf_x = 0.0
    if(is_shift == 0):
      return shift_x, ret_points
    shift_x = ret_points[:,1][-1]
    ret_points[:,1] = ret_points[:,1] - shift_x
    return shift_x, ret_points

def normalized(points, is_normalized = 1):
    ret_points = points.copy()
    norm = 1.0
    if(is_normalized == 0):
      return norm, ret_points
    norm = np.linalg.norm(ret_points[:,1])
    ret_points[:,1] = ret_points[:,1] / norm
    return norm, ret_points

def form_y(points_diff, points_shift,  points_norm):
    ret_points = points_diff.copy()[-1]
    ret_points = ret_points - points_shift
    ret_points = ret_points / points_norm
    return ret_points

# 检查 CUDA 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.5):
        super(LSTM, self).__init__()
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

# Set random seed for reproducibility
torch.manual_seed(0)

# Initialize model, loss function, and optimizer
input_size = 4
hidden_size = 20  # Increased hidden size
output_size = 2
num_layers = 3  # Increased number of layers
dropout = 0.3  # Dropout rate

model = LSTM(input_size, hidden_size, output_size, num_layers, dropout).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 降低学习率


# Training the model
num_epochs = 2000
for epoch in range(num_epochs):
    for data in train_dataset:
        points = data.points
        points_diff = np.diff(points, axis=0)
        _, points_noised = noised(points)# 1 noise(default) 0 clean
        points_shift_x, points_shifted = shifted(points_noised)# 1 shift(default) 0 no shift
        points_shift_x = np.array([points_shift_x])
        points_norm, points_normalized = normalized(points_shifted)# 1 normalize(default) 0 no normalize
        points_norm = np.array([points_norm])
        points_diff_ = np.diff(points_normalized, axis=0)

        # Repeat points_shift_x and points_norm to match the shape of points_diff_
        points_shift_x_repeated = np.repeat(points_shift_x, points_diff_.shape[0], axis=0)
        points_norm_repeated = np.repeat(points_norm, points_diff_.shape[0], axis=0)
        # Expand the dimensions of points_shift_x and points_norm
        points_shift_x_repeated = np.expand_dims(points_shift_x_repeated, axis=1)
        points_norm_repeated = np.expand_dims(points_norm_repeated, axis=1)

        # Concatenate along the feature dimension
        combined_input = np.concatenate((points_diff_[:-1], points_shift_x_repeated[:-1], points_norm_repeated[:-1]), axis=1)

        # Prepare data for LSTM
        X = torch.tensor(combined_input, dtype=torch.float32).unsqueeze(1).to(device)  # Input: (num_points-1, 1, 2)
        y = torch.tensor(points_diff[1:], dtype=torch.float32).to(device)                 # Target: (num_points-1, 2)
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Prediction
model.eval()
with torch.no_grad():
    for data in train_dataset:
        points = data.points
        points_diff = np.diff(points, axis=0)
        _, points_noised = noised(points)# 1 noise(default) 0 clean
        points_shift_x, points_shifted = shifted(points_noised)# 1 shift(default) 0 no shift
        points_shift_x = np.array([points_shift_x])
        points_norm, points_normalized = normalized(points_shifted)# 1 normalize(default) 0 no normalize
        points_norm = np.array([points_norm])
        points_diff_ = np.diff(points_normalized, axis=0)

        # Repeat points_shift_x and points_norm to match the shape of points_diff_
        points_shift_x_repeated = np.repeat(points_shift_x, points_diff_.shape[0], axis=0)
        points_norm_repeated = np.repeat(points_norm, points_diff_.shape[0], axis=0)
        # Expand the dimensions of points_shift_x and points_norm
        points_shift_x_repeated = np.expand_dims(points_shift_x_repeated, axis=1)
        points_norm_repeated = np.expand_dims(points_norm_repeated, axis=1)
        
        # Concatenate along the feature dimension
        combined_input = np.concatenate((points_diff_[:-1], points_shift_x_repeated[:-1], points_norm_repeated[:-1]), axis=1)
        # Prepare data for LSTM
        X = torch.tensor(combined_input, dtype=torch.float32).unsqueeze(1)  # Input: (num_points-1, 1, 2)
        y = torch.tensor(points_diff[1:], dtype=torch.float32)       
        output = model(X[-1].unsqueeze(0))
        y = y[-1].unsqueeze(0)
        Loss = criterion(output, y)
        future_points_diff = output.squeeze().numpy()
        predicted_point = points[-1] + future_points_diff
        plot_prediction(points, points_noised, predicted_point, output_path=f'LSTM_filter/plot/{data.label}_pred.png', loss = Loss.item())
