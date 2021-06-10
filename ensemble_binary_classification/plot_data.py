import pandas
import numpy as np
import math
import seaborn as sns
from sklearn import model_selection
import sklearn.preprocessing as preprocessing
import sklearn.linear_model as linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Plots the distribution of the attributes of the dataset
# Plot the dataset into subplots of 4 columns
def plot_data(original_data):
	fig = plt.figure(figsize=(20,15))

	cols = 4
	rows = math.ceil(float(original_data.shape[1]) / cols)

	for i, column in enumerate(original_data.columns):
		ax = fig.add_subplot(rows, cols, i + 1)

		ax.set_title(column)

		if original_data.dtypes[column] == np.object:
			original_data[column].value_counts().plot(kind="bar", axes=ax)
		else:
			original_data[column].hist(axes=ax)
			plt.xticks(rotation="vertical")

	plt.subplots_adjust(hspace=0.7, wspace=0.3)

	plt.show()
