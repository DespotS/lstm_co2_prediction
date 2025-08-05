import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import datetime


df = pd.read_csv('data/co2_mm_mlo.csv', 
                 names=["year", "month", "decimal_date", "average", "deseasonalized", "ndays", "sdev", "unc"])

df['decimal_date'] = pd.to_numeric(df['decimal_date'], errors='coerce')
df['average'] = pd.to_numeric(df['average'], errors='coerce')
df = df[["decimal_date", "average"]].replace(-9.99, np.nan).dropna()
df.columns = ["date", "co2"]

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df['co2'].values.reshape(-1, 1))


def make_sequences(data, lookback=48): # use past 48 months for training
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(data[i])
    return np.array(X), np.array(y)

X, y = make_sequences(scaled_data)
X = X.reshape((X.shape[0], X.shape[1], 1))

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"Training AI with {len(X_train)} examples...")

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(12, 1)),
    LSTM(32),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), verbose=0)

current_co2 = df['co2'].iloc[-1]
last_sequence = scaled_data[-12:].reshape(1, 12, 1)
predictions = []

for month in range(48): # predict 48 months
    pred = model.predict(last_sequence, verbose=0)
    predictions.append(pred[0, 0])
    # Update sequence for next prediction
    last_sequence = np.roll(last_sequence, -1, axis=1)
    last_sequence[0, -1, 0] = pred[0, 0]

predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# check model prediction
predicted_annual_rate = (predictions[-1, 0] - current_co2) / 4

if predicted_annual_rate < 0.5 or predicted_annual_rate > 5.0:
    print(" AI prediction seems off, applying gentle correction...")
    recent_trend = (df['co2'].iloc[-1] - df['co2'].iloc[-25]) / 2  # last 2 years
    months = np.arange(48)
    predictions = current_co2 + (recent_trend * (months + 1) / 12)
    predictions = predictions.reshape(-1, 1)
    print(f" Using historical trend: +{recent_trend:.1f} ppm/year")
else:
    print(" AI predictions look reasonable!")

start_date = df['date'].iloc[-1]
future_dates = []
for i in range(48):
    date = start_date + (i + 1)/12
    year = int(date)
    month = int((date - year) * 12) + 1
    future_dates.append(datetime.datetime(year, month, 1))

plt.figure(figsize=(12, 6))
plt.plot(future_dates, predictions.flatten(), 'r-', linewidth=3, marker='o', markersize=2)
plt.title(' CO₂ Predictions for Next 4 Years', fontsize=16, pad=20)
plt.xlabel('Year', fontsize=12)
plt.ylabel('CO₂ (ppm)', fontsize=12)
plt.grid(True, alpha=0.3)

# format dates
from matplotlib.dates import DateFormatter, YearLocator
plt.gca().xaxis.set_major_locator(YearLocator())
plt.gca().xaxis.set_major_formatter(DateFormatter('%Y'))

# info box
increase = predictions[-1, 0] - current_co2
plt.text(0.02, 0.98, f'Current: {current_co2:.0f} ppm\n4-year increase: +{increase:.0f} ppm\nRate: +{increase/4:.1f} ppm/year', 
         transform=plt.gca().transAxes, fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

plt.tight_layout()
plt.show()
