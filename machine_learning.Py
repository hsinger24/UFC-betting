# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
#from keras.utils import plot_model

# Calling in data, splitting, and scaling
data = pd.read_csv('mma_data.csv', index_col=0)
data = data[data.result!=-1]
data.drop(['fighter_1', 'fighter_2'], axis = 1, inplace = True)
x, y = data.iloc[:, :-1], data.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Instantiating NN model
model = Sequential()
model.add(Dense(16, input_dim=x_train_scaled.shape[1],
                activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

# Fitting and testing model
model.fit(x=x_train_scaled, y=y_train, epochs=200, batch_size=64, verbose=0)
test_results = model.evaluate(x = x_test_scaled, y = y_test, verbose=0)
print("Test Accuracy = {}".format(test_results[1]))
