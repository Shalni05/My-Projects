''' Churn Modelling Using an Artificial Neural Network
    The code is from Udemy's Deep Learning A-Z™: Hands-On Artificial Neural Networks course
'''

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Function to take care of the encoding.
def categorical_encoder(data, index): # Takes a numpy array and the index of the field to be encoded.
    label_encoder = LabelEncoder()
    data[:, index] = label_encoder.fit_transform(data[:, index])
    return(data)
	
def split_date(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    return(X_train, X_test, y_train, y_test)
	
def feature_scaler(data):
    sc = StandardScaler()
    return(sc.fit_transform(data))

def dummy_variable_maker(data, index):
    onehotencoder = OneHotEncoder(categorical_features = [index])
    data = onehotencoder.fit_transform(data).toarray()
    return(data)
	
# Importing the dataset

	

def train_model(filename):
	dataset = pd.read_csv(filename)
	# Taking all features that drive the Y-variable.
	X = dataset.iloc[:, 3:13].values 
	y = dataset.iloc[:, 13].values

	# The Categorical variables are in the form of Strings - they need to be encoded before inputting them into a Neural Net.
	# Dependent variable has only two possible values 1s and 0s. 
	# Use the LabelEncoder and OneHotEncoder to encode the driver variables.

	# Encoding the field Country which is index 1
	X = categorical_encoder(X,1)

	# Encoding gender
	X = categorical_encoder(X, 2)

	# Using Dummy Variables to fix the relational ordering of Geography field.
	X = dummy_variable_maker(X,1)
	X = X[:, 1:] # Taking all columns except the first one.

	# Split the dataset
	X_train, X_test, y_train, y_test = split_date(X,y)

	# Feature scaling is absolutely necessarily as there is a lot of computation involved -- 
	# computationally intensive calculations require parallel computing and feature scaling makes the computation easier. 
	# Therefore, one independent variable cannot dominate another.

	X_train = feature_scaler(X_train)
	X_test = feature_scaler(X_test)


	# Building the neural network

	classifier = Sequential()
	classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
	classifier.add(Dense(kernel_initializer="uniform", units=6, activation="relu"))
	classifier.add(Dense(kernel_initializer="uniform", units=1, activation="sigmoid"))

	# Training the model 

	classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])
	classifier.fit(X_train, y_train, batch_size=10, epochs=50)



	# Testing the model

	# Can't mix binary variables (y_test) and continuous variabels (y_pred). 
	# We will convert the probabilities into a binary form too
	y_pred = classifier.predict(X_test)
	y_pred = (y_pred > 0.5)
	cm = confusion_matrix(y_test, y_pred)
	
	

	return cm, accuracy_score(y_test, y_pred)


