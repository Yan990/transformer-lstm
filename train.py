from math import sqrt
import numpy as np
from pandas import read_excel, DataFrame, concat
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, LSTM, Add, LayerNormalization, MultiHeadAttention, Dropout, GlobalAveragePooling1D
from tensorflow.keras.models import Model

# --- 1. Architecture Definitions ---
def encoder_block(x, d_model, num_heads):
    attn_out = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
    x = Add()([x, attn_out])
    x = LayerNormalization(epsilon=1e-6)(x)
    ffn_out = Dense(d_model * 4, activation='relu')(x)
    ffn_out = Dense(d_model)(ffn_out)
    x = Add()([x, ffn_out])
    x = LayerNormalization(epsilon=1e-6)(x)
    return x

def decoder_block(x, d_model):
    lstm_out = LSTM(d_model, return_sequences=True)(x)
    x = Add()([x, lstm_out])
    x = LayerNormalization(epsilon=1e-6)(x)
    ffn_out = Dense(d_model * 4, activation='relu')(x)
    ffn_out = Dense(d_model)(ffn_out)
    x = Add()([x, ffn_out])
    x = LayerNormalization(epsilon=1e-6)(x)
    return x

def build_hybrid_transformer_lstm(n_seconds, n_features, num_layers=3, d_model=64, num_heads=4):
    inputs = Input(shape=(n_seconds, n_features))
    x = Dense(d_model)(inputs)
    
    decoder_outputs = []
    current_encoder_out = x
    
    for _ in range(num_layers):
        current_encoder_out = encoder_block(current_encoder_out, d_model, num_heads)
        dec_out = decoder_block(current_encoder_out, d_model)
        decoder_outputs.append(dec_out)
        
    combined_decoder = Add()(decoder_outputs) if num_layers > 1 else decoder_outputs[0]
    pooled = GlobalAveragePooling1D()(combined_decoder)
    
    mlp_out = Dense(d_model // 2, activation='relu')(pooled)
    outputs = Dense(1)(mlp_out)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='mean_absolute_error', optimizer='adam')
    return model

# --- 2. Data Preparation ---
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    cols.append(df.shift(-n_out+1))
    names += [('var%d(t+%d)' % (j+1, n_out)) for j in range(n_vars)]
    agg = concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# Parameters
n_seconds = 10 # <-- NOTE: Increase this to capture temporal dependencies (e.g., 45)!
n_features = 7 

# Load dataset
dataset = read_excel('./File/FL2409JJ9070-004-4.xlsx', header=0, index_col=0)
values = dataset.values
y_max = np.max(values[:, -1])
y_min = np.min(values[:,-1])

# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

# Reframe as supervised learning
reframed = series_to_supervised(scaled, n_seconds, 1)
values = reframed.values.astype('float32')

# Split train/test
train = values[:25000, :]
test = values[25000:, :]

# Split into input and outputs
n_obs = n_seconds * n_features
train_X, train_y = train[:, :n_obs], train[:, -1]
test_X, test_y = test[:, :n_obs], test[:, -1]

# Reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], n_seconds, n_features))
test_X = test_X.reshape((test_X.shape[0], n_seconds, n_features))

print(f"Train X shape: {train_X.shape}, Train Y shape: {train_y.shape}")

# --- 3. Model Training ---
# Build the model using our custom function
model = build_hybrid_transformer_lstm(
    n_seconds=n_seconds, 
    n_features=n_features, 
    num_layers=2,   # 'N' in your diagram
    d_model=64,     # Dimension of hidden layers 'C'
    num_heads=4
)

# Fit network
history = model.fit(
    train_X, train_y, 
    epochs=50, # Reduced for testing, adjust as needed
    batch_size=72, 
    validation_data=(test_X, test_y), 
    verbose=2, 
    shuffle=False
)

# --- 4. Evaluation & Plotting ---
pyplot.figure(figsize=(10, 8))

# Plot training history
pyplot.subplot(2, 1, 1)
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.title('Model Loss')
pyplot.legend()

# Make a prediction
yhat = model.predict(test_X)

# Invert scaling
yhat_y = yhat * (y_max - y_min) + y_min
t_y = test_y * (y_max - y_min) + y_min

# Plot predictions
pyplot.subplot(2, 1, 2)
pyplot.plot(yhat_y[:,0], label='Predicted', alpha=0.7)
pyplot.plot(t_y, label='Actual', alpha=0.7)
pyplot.title('Moisture Content Prediction')
pyplot.legend()
pyplot.tight_layout()
pyplot.show()

# Calculate MAE
mae = mean_absolute_error(t_y, yhat_y)
print('Test MAE: %.4f' % mae)