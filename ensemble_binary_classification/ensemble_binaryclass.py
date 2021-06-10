import pandas
import numpy as np
import math
import seaborn as sns
from sklearn import model_selection
import sklearn.preprocessing as preprocessing
import sklearn.linear_model as linear_model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import sklearn.metrics
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from data_reader import *
from plot_data import *




path="adult.data"
names = ['age','workclass','fnlwgt','education','education-num','marital-status',
         'occupation','relationship','race','sex','capital-gain','capital-loss',
         'hours-per-week','native-country','class']


# Read the dataset from file
# encoded_data is the integer encoded copy
# original_data is the original copy
encoded_data, original_data = read_data(path , names)

# Add the following line to plot the code
#plot_data(original_data)


# The attribute names
train_cols = ['age','workclass','fnlwgt','education-num','marital-status',
         'occupation','relationship','race','sex','capital-gain','capital-loss',
         'hours-per-week','native-country']



X=encoded_data[train_cols].values
Y=encoded_data['class'].values

# Splitting the dataset (70% and 30%)
X_train, X_test, y_train, y_test = train_test_split(X,Y, train_size=0.70, test_size=0.30)



#################### MLP  ####################

# Training a multilayer perceptron on the training dataset
print ("Multilayer Perceptron Accuracy on Training Data:")

dt =   MLPClassifier(alpha=1, hidden_layer_sizes=(25, 4))
model6 = dt.fit(X_train,y_train)

# Getting the predictions of MLP on test dataset
preidction6= model6.predict(X_test)

# Getting the class probabilities of MLP on test dataset
predict_prob6 = model6.predict_proba(X_test)

# The confusion matrix of MLP on test data
cfm_decisiontreeboost= sklearn.metrics.confusion_matrix(y_test,preidction6)

# The accuryacy of MLP on test data
accu_decisiontreeboost= sklearn.metrics.accuracy_score(y_test, preidction6)

print('confusion_matrix=\n', cfm_decisiontreeboost)

print('accuracy_score=',accu_decisiontreeboost)



########## Logistic Regression  (LR) ##########

# Training s ogictic regression model on training dataset
print ("Logistic regression Accuracy on Training Data:")

model = linear_model.LogisticRegression()
model7 = model.fit(X_train,y_train)

# Getting the predictions of LR on test dataset
preidction7 = model7.predict(X_test)

# Getting the class probabilities of LR on test dataset
predict_prob7 = model7.predict_proba(X_test)


# The confusion matrix of LR on test data
cfm_logistic= sklearn.metrics.confusion_matrix(y_test, preidction7)

# The accuracy of LR on test data
accu_logistic= sklearn.metrics.accuracy_score(y_test, preidction7)

print('confusion_matrix=\n', cfm_logistic)

print('accuracy_score=',accu_logistic)



############## Naive Bayes (NB) ####################

print ("Naive Bayes Accuracy on Training Data:")

# Training a Naive Bayes on the training data
model = GaussianNB()
model8 = model.fit(X_train,y_train)

# Getting the predictions of NB on test dataset
preidction8 = model8.predict(X_test)

# Getting the class probabilities of NB on test dataset
predict_prob8 = model8.predict_proba(X_test)

# The confusion matrix of NB on test data
cfm_gradient= sklearn.metrics.confusion_matrix(y_test,preidction8)

# The accuracy of NB on test data
accu_gradient= sklearn.metrics.accuracy_score(y_test, preidction8)

print('confusion_matrix=\n', cfm_gradient)

print('accuracy_score=',accu_gradient)


# The meta-data with class probabilities
d = {
   'c6':preidction6,
   'p6': predict_prob6[:,0],
   'c7':preidction7,
   'p7': predict_prob7[:,0],
   'c8':preidction8,
   'p8': predict_prob8[:,0],

  }



# The meta-data with class probabilities
e = {
   'c6':preidction6,
   'c7':preidction7,
   'c8':preidction8,
  }


# Create data-frames for the metadata
df=pandas.DataFrame(data=d)
ef=pandas.DataFrame(data=e)


################## Stacking Classifiers #############################
# Training the meta-level classifier 
base = RandomForestClassifier()

# this will include classes and class probabilities in meta-data
classifier_full = BaggingClassifier(base_estimator=base, n_estimators=100)
clf_full=classifier_full.fit(df,y_test)


# this will only include classes in meta-data
classifier_half = BaggingClassifier(base_estimator=base, n_estimators=100)
clf_half=classifier_half.fit(ef,y_test)




# Get test dataset
test_encoded_data, original_data = read_data("adult.test" , names)


X=test_encoded_data[train_cols].values
Y=test_encoded_data['class'].values

# Get the predictions of the base classifiers for test data
# Predictions of MLP
c6 = model6.predict(X)

# Predictions of LR
c7 = model7.predict(X)

# Predictions of NB
c8 = model8.predict(X)


# Get the class probabilities for test data from the base classifiers
# Predictions of MLP
p6= model6.predict_proba(X)

# Predictions of LR
p7 = model7.predict_proba(X)

# Predictions of NB
p8 = model8.predict_proba(X)


#### Meta data: Prediction with class probabilities ####

f = {
   'c6':c6,
   'p6': p6[:,0],
   'c7':c7,
   'p7': p7[:,0],
   'c8':c8,
   'p8': p8[:,0],

  }

# Final predictions of the Stacking (With Class Probabilites) algorithm
test_df_full=pandas.DataFrame(data=f)
pred_full = clf_full.predict(test_df_full)


# Performance measurements
print("Combined Algorithm (With Class Probabilities) Confusion Table on Test DataSet:")
 
print(sklearn.metrics.confusion_matrix(Y,pred_full))

print(sklearn.metrics.accuracy_score(Y, pred_full))

print('F1 score = ', sklearn.metrics.f1_score(Y,pred_full))

print('Precision score=',sklearn.metrics.precision_score(Y,pred_full))

print('Recall score=',sklearn.metrics.recall_score(Y,pred_full))



#### Meta data: Prediction  ####

e1 = {
   'c6':c6,
   'c7':c7,
   'c8':c8,
  }

ef=pandas.DataFrame(data=e1)

# Final predictions of the Stacking (With Class Probabilites) algorithm
pred_half = clf_half.predict(ef)

# Performance measurements
print("Combined Algorithm (Without Class Probabilities) Confusion Table on Test DataSet:")

print(sklearn.metrics.confusion_matrix(Y,pred_half))

accu_logistic= sklearn.metrics.accuracy_score(Y, pred_half)

print(accu_logistic)

print('F1 score = ', sklearn.metrics.f1_score(Y,pred_half))

print('Precision score=',sklearn.metrics.precision_score(Y,pred_half))

print('Recall score=',sklearn.metrics.recall_score(Y,pred_half))



# Predictions of the base classes on the real test dataset
print("NN Accuract on Test DataSet:")
print(sklearn.metrics.accuracy_score(Y, model6.predict(X)))

print("Logical Regression Accuracy on Test DataSet:")
print(sklearn.metrics.accuracy_score(Y, model7.predict(X)))

print("Naive Bayes Accuracy on Test DataSet:")
print(sklearn.metrics.accuracy_score(Y, model8.predict(X)))
