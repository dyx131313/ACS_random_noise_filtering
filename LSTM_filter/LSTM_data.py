import numpy as np

def gen_points(x, expression):
	y = eval(expression)
	return np.column_stack((x,y))

class dataset:
	def gen_data(self, start = 0.5, stop = 3.5, points = 20, expression = "x"):
		self.x = np.linspace(start,stop,points)
		# self.points = []
		self.points = gen_points(self.x, expression)
		self.displacements = np.diff(self.points, axis=0)
	def __init__(self,label = "trajectory", expression = "x"):
		self.label = label
		self.expression = expression
		self.gen_data(expression = self.expression)

train_dataset = [dataset(expression = "np.sin(x)",label = "sin"),
				 dataset(expression = "np.cos(x)",label = "cos"),
				 dataset(expression = "x",label = "liner"),
				 dataset(expression = "x * x -2 *x ",label = "quadratic"),
                 dataset(expression = " 1/x ",label = "verse"),
                 dataset(expression = "2 / x", label = "verse2"),
                 dataset(expression = "10 / x", label = "verse3")
				#  dataset(expression = "np.sigmoid(x-2)",label = "sigmoid")
				 ]