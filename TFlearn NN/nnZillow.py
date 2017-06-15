# import dependencies
import pandas as pd
import numpy as np
import tflearn as tf
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# read in data
data = pd.read_csv('./clean_data_zillow.csv', index_col=0)

# drop parcelid from the feature set during training.
data = data.drop(['parcelid', 'transactiondate'], axis=1)

# split data into samples and labels
X = data.iloc[:,1:]
y = data.iloc[:,:1]
# print('X size: ', X.shape)
# print('y size: ', y.shape)

# now we want to scale the data, first we start with X
scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)

# now we scale y
scaler1 = preprocessing.MinMaxScaler()
y_scaled = scaler1.fit_transform(y)

# split data into training and testing sets, with testing size equal to 20% of total data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# input layer
net = tf.input_data(shape=[None, X_train.shape[1]])

# hidden layers
net = tf.fully_connected(net, 32, activation='relu')
net = tf.fully_connected(net, 64, activation='relu')
net = tf.dropout(net, 0.5)
net = tf.fully_connected(net, 64, activation='relu')
net = tf.fully_connected(net, 32, activation='relu')

# output layer
net = tf.fully_connected(net, 1, activation='linear')

net = tf.regression(net, optimizer='adam', loss='mean_square', metric='R2', learning_rate=0.0001)

net = tf.DNN(net, tensorboard_verbose=0)

# fit the network
net.fit(X_train, y_train, n_epoch=10, validation_set=0.1, show_metric=True, batch_size=128, shuffle=True, snapshot_epoch=True)

# save model
# net.save("zillow_tflearn_nn.tfl")

# try on the test set, predict on 20 samples
acc = []
for i in range(y_test.shape[0]):
	# make a random number in range of the test set
	sample_num = np.random.randint(0, X_test.shape[0])

	# make a prediction on that random sample
	pred = net.predict([X_test[sample_num]])

	# actual label for that test data
	actual_val = y_test[sample_num]

	# calculate how far off our prediction is from the actual value
	error = actual_val - pred
	acc.append(error)

	print("Actual value: {} \t Predicted Value: {} \t Accuracy: {}".format(actual_val, pred, acc[i]))

# calculate and print the avg error.
avg_err = sum(acc)/len(acc)
print("Average Error: {}".format(avg_err))







