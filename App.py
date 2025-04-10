import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
import streamlit as st

# Load dataset
data = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv")
data["Temp"] = data["Temp"].astype(float)

# Sequence creator
def create_sequences(series, window=7):
    X, y = [], []
    for i in range(len(series) - window):
        X.append(series[i:i+window])
        y.append(series[i+window])
    return np.array(X), np.array(y)

# Preprocess
data_values = data["Temp"].values
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data_values.reshape(-1, 1)).flatten()
X, y = create_sequences(data_scaled)

# Build model
model = Sequential([
    Dense(64, activation='relu', input_shape=(7,)),
    Dense(32, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=50, batch_size=16, verbose=0)

# Streamlit app
st.title("🌡️ Temperature Forecasting Dashboard")
st.write("Enter the last 7 days of minimum temperatures to predict the 8th day temperature.")

user_input = []
for i in range(7):
    temp = st.number_input(f"Day {i+1} Temperature (°C)", value=15.0, step=0.1)
    user_input.append(temp)

if st.button("Predict 8th Day Temperature"):
    input_array = scaler.transform(np.array(user_input).reshape(-1, 1)).flatten().reshape(1, -1)
    prediction_scaled = model.predict(input_array)
    prediction = scaler.inverse_transform(prediction_scaled)[0][0]
    st.success(f"🌞 Predicted Temperature for Day 8: {prediction:.2f}°C")
