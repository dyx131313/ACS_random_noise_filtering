import matplotlib.pyplot as plt

# Visualization

def plot_prediction(points, points_noised, predicted_point, label = "default", output_path = None, loss = 0.0):
        plt.figure(figsize=(8, 6))
        plt.plot(points[:, 0], points[:, 1], 'yo-', label='Original points')
        plt.plot(points_noised[:, 0], points_noised[:, 1], 'bo-', label='Observation points')
        plt.scatter(predicted_point[0], predicted_point[1], color='red', label='Predicted point', marker='x', s=100)
        plt.title(f'Prediction of {label} using LSTM\nLoss: {loss:.4f}')
        plt.xlabel(f'T-axis')
        plt.ylabel(f'{label}-axis')
        plt.legend()
        plt.grid(True)

        if output_path:
            plt.savefig(output_path)  # 保存图像到指定路径
        else:
            plt.show()  # 显示图像