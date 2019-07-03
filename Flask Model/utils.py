# imports
import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

from bson import dumps
from flask import jsonify


import keras
# The sequential module is used to initialize our ANN
from keras.models import Sequential
# The dense module is required to build the layers of our ANN
from keras.layers import Dense
from keras.models import load_model



def read_data(file_path):
    if file_path.endswith('.json'):
        return pd.read_json(file_path)
    elif file_path.endswith('.csv'):
        return pd.read_csv(file_path, index_col=0)

def get_corresponding_columns(dataset, col_names):
    return [dataset.columns.get_loc(c) for c in dataset.columns if c in col_names]

def create_feature_matrix(df, columns):
    cols = get_corresponding_columns(df, columns)
    return pd.DataFrame(df.iloc[:,cols])

def target_variable_selector(df, column):
    index = get_corresponding_columns(df, column)
    return pd.DataFrame(df.iloc[:, index].values)

def split_data(X, y, test_size):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = 0)
    return(X_train, X_test, y_train, y_test)

def feature_scaler(data):
    sc = StandardScaler()
    return(pd.DataFrame(sc.fit_transform(data)))

def create_ann(X_train):
    classifier = Sequential()
    # Input layer
    classifier.add(Dense(kernel_initializer="uniform", units=len(X_train.columns), activation="relu", input_dim=len(X_train.columns)))
    # Second layer - Hidden layer
    classifier.add(Dense(kernel_initializer="uniform", units=int((len(X_train.columns)+1)/2), activation="relu"))
    # Output layer
    classifier.add(Dense(kernel_initializer="uniform", units=1, activation="sigmoid"))
    # compile classifier
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])
    return classifier

def read_and_wrangle(file_path):
	dataset = read_data(file_path)
	fm = create_feature_matrix(dataset, dataset.columns[2:12]) # Hardcoded
	targ_var = targ = target_variable_selector(dataset, 'Exited') # Hardcoded
	# Applying variable encoding to factorize categorical variables and then one-hot encoding to create dummies
	return (pd.get_dummies(fm, drop_first=True), targ_var)

def apply_model(experiment, X_train, y_train):
	model = ''
	if experiment['type'] == "ann":
		model = create_ann(X_train)

	trained_model = model.fit(X_train, y_train, batch_size=10, epochs=20)
	return trained_model

def train_model(file_path, experiment):
	train_pc = experiment['train_split']
	test_pc = (100 - int(train_pc))/100
	X,y = read_and_wrangle(file_path)
	X_train, X_test, y_train, y_test = split_data(X,y, test_pc)

	LOG.info('Train and test splits created!')

	X_train = feature_scaler(X_train)
	LOG.info('Feature scaling on x_train completed.')
	#X_test = feature_scaler(X_test)

	trained_model = apply_model(experiment, X_train, y_train)
	trained_model.model.save("model.h5")
	LOG.info('Model trained and saved!')
	return("Model trained and saved!")

def cm_results(cm):
	tp = cm[0][0]
	fn = cm[0][1]
	fp = cm[1][0]
	tn = cm[1][1]
	error_rate = (tp/np.sum(cm))*100 
	accuracy = ((tp+tn)/np.sum(cm))*100
	sensitivity = (tp/(tp+fn))*100
	specificity = (tn/(fp+tn))*100
	precision = (tp/(tp+fp))*100
	return(dict({"error_rate":error_rate, "accuracy":accuracy, "sensitivity":sensitivity,
		"specificity":specificity,"precision":precision}))


def test_model(file_path, experiment):
	model = load_model('model.h5')
	train_pc = experiment['train_split']
	test_pc = (100 - int(train_pc))/100

	X,y = read_and_wrangle(file_path)
	X_train, X_test, y_train, y_test = split_data(X,y, test_pc)

	#X_train = feature_scaler(X_train)
	X_test = feature_scaler(X_test)
	# test the model on the x test data
	y_pred = model.predict(X_test)

	y_pred = (y_pred > 0.5)
	cm = confusion_matrix(y_test, y_pred)
	return cm_results(cm)