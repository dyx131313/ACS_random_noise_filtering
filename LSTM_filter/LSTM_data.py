import numpy as np
import csv

def gen_points(x, expression):
    y = eval(expression)
    return np.column_stack((x, y))

class dataset:
    def __init__(self, label="trajectory", expression="x", csv_file=None):
        self.label = label
        self.expression = expression
        if csv_file:
            self.load_from_csv(csv_file)
        else:
            self.gen_data(expression=self.expression)

    def gen_data(self, start=0.5, stop=3.5, points=20, expression="x"):
        self.x = np.linspace(start, stop, points)
        self.points = gen_points(self.x, expression)
        self.displacements = np.diff(self.points, axis=0)
        self.save_to_csv(f'dataset/{self.label}_points.csv')

    def load_from_csv(self, csv_file):
        with open(csv_file, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            self.points = []
            for row in reader:
                if row[0]:  # Check if the row is not empty
                    self.points.append([float(row[0]), float(row[1])])
            self.points = np.array(self.points)
            self.displacements = np.diff(self.points, axis=0)

    def save_to_csv(self, csv_file):
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['x', 'y'])
            for point in self.points:
                writer.writerow(point)

# 生成并保存数据
train_dataset = [
    dataset(expression="np.sin(x)", label="sin"),
    dataset(expression="np.cos(x)", label="cos"),
    dataset(expression="x", label="liner"),
    dataset(expression="x * x - 2 * x", label="quadratic"),
    dataset(expression="1 / x", label="verse"),
    dataset(expression="2 / x", label="verse2"),
    dataset(expression="10 / x", label="verse3")
]
# # 示例：从CSV文件中加载数据
# train_dataset = [
#     dataset(csv_file='path/to/sin_points.csv', label='sin'),
#     dataset(csv_file='path/to/cos_points.csv', label='cos'),
#     dataset(csv_file='path/to/liner_points.csv', label='liner'),
#     dataset(csv_file='path/to/quadratic_points.csv', label='quadratic'),
#     dataset(csv_file='path/to/verse_points.csv', label='verse'),
#     dataset(csv_file='path/to/verse2_points.csv', label='verse2'),
#     dataset(csv_file='path/to/verse3_points.csv', label='verse3')
# ]

#  # Save points to CSV
# if(epoch == 0):
#     with open(f'LSTM_filter/dataset/{data.label}_train_points.csv', mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(['x', 'y'])
#         for point in points:
#             writer.writerow(point)