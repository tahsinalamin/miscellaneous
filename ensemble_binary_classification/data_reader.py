import pandas
import numpy as np
import math
import seaborn as sns
#import statsmodels as sm
from sklearn import model_selection
import sklearn.preprocessing as preprocessing
import sklearn.linear_model as linear_model
from sklearn.model_selection import train_test_split



# Encodes the data into integers
def number_encode_features(df):
    result = df.copy()
    encoders = {}

    for column in result.columns:
        if result.dtypes[column] == np.object:
            encoders[column] = preprocessing.LabelEncoder()
            result[column] = encoders[column].fit_transform(result[column])
    return result, encoders


# Reads dataset from url or file 
def read_data(path, names):
    # path should contain the url or file path
    # names should be a list of the attribute names
	dataframe = pandas.read_csv(path,names=names)
	array=dataframe.values
	encoded_data, _ = number_encode_features(dataframe)
    # we are returing the encoded and the original dataframe
	return encoded_data, dataframe
