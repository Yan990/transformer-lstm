from math import sqrt

from numpy import concatenate

from matplotlib import pyplot

from pandas import read_excel

from pandas import DataFrame

from pandas import concat

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import  Dropout, Embedding, Bidirectional

import numpy as np

from tensorflow import keras

# convert series to supervised learning

def series_to_supervised(data, n_in=1, n_out=30, dropnan=True):

    n_vars = 1 if type(data) is list else data.shape[1]

    df = DataFrame(data)

    cols, names = list(), list()

    # input sequence (t-n, ... t-1)

    for i in range(n_in, 0, -1):

        cols.append(df.shift(i))

        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

    # forecast sequence (t, t+1, ... t+n)

    # for i in range(0, n_out):

    cols.append(df.shift(-n_out+1))

        # if i == 0:

            # names += [('var%d(t)' % (j+1)) for j in range(n_vars)]

        # else:

    names += [('var%d(t+%d)' % (j+1, n_out)) for j in range(n_vars)]

    # put it all together

    agg = concat(cols, axis=1)

    agg.columns = names

    # drop rows with NaN values

    if dropnan:

        agg.dropna(inplace=True)

    return agg



# load dataset

dataset = read_excel('./File/FL2409JJ9070-004-4.xlsx', header=0, index_col=0)

values = dataset.values

y_max=np.max(values[:, -1])

y_min=np.min(values[:,-1])

# integer encode direction

# encoder = LabelEncoder()

n_seconds = 1

n_features = 7

# values[:,4] = encoder.fit_transform(values[:,4])

# reframed = series_to_supervised(values, n_seconds, 30)

values = reframed.values

scaler = MinMaxScaler(feature_range=(0, 1))

scaled = scaler.fit_transform(values)

# specify the number of lag hours



# frame as supervised learning



# split into train and test sets

values=scaled







# ensure all data is float

values = values.astype('float32')

# normalize features









train = values[:25000, :]

test = values[25000:, :]

# reframed = series_to_supervised(values, n_seconds, 1)

print(reframed.shape)

# split into input and outputs

n_obs = n_seconds * n_features

train_X, train_y = train[:, :n_obs], train[:, -1]

test_X, test_y = test[:, :n_obs], test[:, -1]

print(train_X.shape, len(train_X), train_y.shape)

# reshape input to be 3D [samples, timesteps, features]

train_X = train_X.reshape((train_X.shape[0], n_seconds, n_features))

test_X = test_X.reshape((test_X.shape[0], n_seconds, n_features))

print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)



# design network

model = Sequential()

model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))



model.add(Dense(1))



model.compile(loss='mean_absolute_error', optimizer='adam')

# fit network

history = model.fit(train_X, train_y, epochs=100, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)

# plot history

pyplot.subplot(221)

pyplot.plot(history.history['loss'], label='train')

pyplot.plot(history.history['val_loss'], label='test')



pyplot.legend()

pyplot.show()



# make a prediction

yhat = model.predict(test_X)





yhat_y = yhat * (y_max - y_min) + y_min

t_y = test_y * (y_max - y_min) + y_min

# print(y_max)

# print(y_min)

pyplot.subplot(222)

pyplot.plot(yhat_y[:,0], label='test_predict')

pyplot.plot(t_y,label='test_y')

# print(yhat_y)

# print(t_y)

# test_X = test_X.reshape((test_X.shape[0], n_seconds*n_features))

# # invert scaling for forecast

# inv_yhat = concatenate((yhat, test_X[:, -8:-1]), axis=1)

# inv_yhat = scaler.inverse_transform(inv_yhat)

# inv_yhat = inv_yhat[:,0]

# # invert scaling for actual

# test_y = test_y.reshape((len(test_y), 1))

# inv_y = concatenate((test_y, test_X[:,-8:-1]), axis=1)

# inv_y = scaler.inverse_transform(inv_y)

# inv_y = inv_y[:,0]

# calculate RMSE



# rmse = sqrt(mean_squared_error(t_y, yhat_y))

mae = mean_absolute_error(t_y, yhat_y)

# mae1=mean_absolute_error(t_y, yhat_y)

# mae2=mean_absolute_error(test_y, yhat)

print('Test MAE: %.4f' % mae) 