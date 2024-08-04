import matplotlib.pyplot as plt

# Visualization

def plot_prediction(points, points_noised, predicted_point, output_path = None, loss = 0.0):
        plt.figure(figsize=(8, 6))
        plt.plot(points[:, 0], points[:, 1], 'yo-', label='Original points')
        plt.plot(points_noised[:, 0], points_noised[:, 1], 'bo-', label='Observation points')
        plt.scatter(predicted_point[0], predicted_point[1], color='red', label='Predicted point', marker='x', s=100)
        plt.title(f'Prediction of the next point on 2D plane using LSTM\nLoss: {loss:.4f}')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.legend()
        plt.grid(True)

        if output_path:
            plt.savefig(output_path)  # 保存图像到指定路径
        else:
            plt.show()  # 显示图像