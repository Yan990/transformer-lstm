import numpy as np
import pandas as pd
from matplotlib import pyplot
from pandas import read_excel, DataFrame, concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from tensorflow import keras

# 1. Convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    cols.append(df.shift(-n_out+1))
    names += [('var%d(t+%d)' % (j+1, n_out)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# 2. Parameters (MUST match your training parameters)
n_seconds = 10  # Ensure this matches the sequence length used during training!
n_features = 7

# 3. Load and prepare dataset
dataset = read_excel('./File/FL2409JJ9070-004-4.xlsx', header=0, index_col=0)
raw_values = dataset.values

# Save min/max for the target variable (assuming target is the last column)
y_max = np.max(raw_values[:, -1])
y_min = np.min(raw_values[:, -1])

# Reframe for supervised learning
reframed = series_to_supervised(raw_values, n_in=n_seconds, n_out=1)
values = reframed.values

# Extract the true unscaled y values for our specific test slice
# Note: Because of dropnan in series_to_supervised, row indices shift slightly. 
y_test_true = values[0:1000, -1]

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
scaled = scaled.astype('float32')

# 4. Extract test slice
test = scaled[0:1000, :]
n_obs = n_seconds * n_features

# Split into X and y
test_X, test_y = test[:, :n_obs], test[:, -1]

# Reshape input to 3D [samples, timesteps, features] for the Transformer-LSTM
test_X = test_X.reshape((test_X.shape[0], n_seconds, n_features))

# 5. Load Model & Predict
# Replace with your actual saved model name
model = keras.models.load_model("Hybrid_Transformer_LSTM.keras") 

print(f"Test X shape: {test_X.shape}")
yhat = model.predict(test_X)
print(f"Predictions shape: {yhat.shape}")
print(f"True Y shape: {y_test_true.shape}")

yhat_y = yhat * (y_max - y_min) + y_min


data = pd.DataFrame({'true': y_test_true, 'pre': yhat_y[:, 0]})
data.to_excel('13000-14000_predictions.xlsx', index=False)


pyplot.figure(figsize=(10, 5))
pyplot.subplot(111)

pyplot.plot(yhat_y[:, 0], linewidth=0.8, color='red', label='Predict')
pyplot.plot(y_test_true, linewidth=0.8, color='blue', label='True', alpha=0.7)

ax = pyplot.gca()
ax.spines['bottom'].set_linewidth(0.5)
ax.spines['left'].set_linewidth(0.5)
ax.spines['right'].set_linewidth(0.5)
ax.spines['top'].set_linewidth(0.5)

pyplot.title("Moisture Content: True vs Predicted")
pyplot.legend()
pyplot.tight_layout()
pyplot.show()


mae = mean_absolute_error(y_test_true, yhat_y[:, 0])
print('Test MAE: %.4f' % mae)