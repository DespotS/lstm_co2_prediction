import pandas as pd
from autots import AutoTS
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# === Load and clean CSV ===
df = pd.read_csv("data/co2_mm_mlo.csv", sep=",", comment="#")
df["average"] = pd.to_numeric(df["average"], errors="coerce")
df["decimal date"] = pd.to_numeric(df["decimal date"], errors="coerce")
df = df[df["average"] > 0]
df = df[["decimal date", "average"]].copy()
df.columns = ["date_decimal", "CO2"]

# === Convert decimal year to datetime (monthly) ===
def decimal_year_to_month(decimal_year):
    year = int(decimal_year)
    rem = decimal_year - year
    month = int(round(rem * 12)) + 1
    if month > 12:
        month = 12
    return datetime(year, month, 1)

df["date"] = df["date_decimal"].apply(decimal_year_to_month)
df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
df_final = df[["date", "CO2"]].copy()

# === AutoTS Forecasting ===
model = AutoTS(
    forecast_length=48,  # 4 years
    frequency='M',
    ensemble="simple",
    model_list=["ARIMA", "ETS", "Prophet", "LastValueNaive", "DatepartRegression"],
    transformer_list="superfast",
    max_generations=4,
    num_validations=2,
    validation_method="backwards"
)
model = model.fit(df_final, date_col="date", value_col="CO2", id_col=None)
prediction = model.predict()
forecast_df = prediction.forecast

# === Calculate rate and summary for legend ===
last_2024 = df_final[df_final["date"].dt.year == 2024]["CO2"].iloc[-1]
last_forecast = forecast_df["CO2"].iloc[-1]
forecast_end_year = forecast_df.index[-1].year
years_forecasted = forecast_end_year - 2024
ppm_rate = round((last_forecast - last_2024) / years_forecasted, 2)

# === Plot ===
plt.figure(figsize=(12, 6))
df_plot = df_final[df_final["date"] >= datetime(1980, 1, 1)]
plt.plot(df_plot["date"], df_plot["CO2"], label="Historical CO₂")
plt.plot(forecast_df.index, forecast_df["CO2"], label="Forecast CO₂")

# Clean x-axis (years only)
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.xticks(rotation=45)

# Add labels
plt.title("CO₂ Forecast with AutoTS (From 1970)")
plt.xlabel("Year")
plt.ylabel("CO₂ ppm")
plt.legend(
    title=f"2024 CO₂: {last_2024:.2f} ppm\n"
          f"{forecast_end_year} CO₂: {last_forecast:.2f} ppm\n"
          f"↑ {ppm_rate} ppm/year"
)
plt.grid(True)
plt.tight_layout()
plt.show()
print(model.best_model)