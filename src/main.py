from data_processing import process_data
from model_training import train_model
from backtest import backtest

sp500 = process_data()
predictors = ["Close", "Volume", "Open", "High", "Low"]

model = train_model(sp500, predictors)

predictions, precision = backtest(model, sp500, predictors)

horizons = [2, 5, 60, 250, 1000]
new_predictors = []

for horizon in horizons:
    rolling_averages = sp500.rolling(horizon).mean()

    ratio_column = f"Close_Ratio_{horizon}"
    sp500[ratio_column] = sp500["Close"] / rolling_averages["Close"]

    trend_column = f"Trend_{horizon}"
    sp500[trend_column] = sp500.shift(1).rolling(horizon).sum()["Target"]

    new_predictors += [ratio_column, trend_column]

sp500 = sp500.dropna()

model = train_model(sp500, new_predictors)

predictions, precision = backtest(model, sp500, new_predictors)

print(predictions, predictors)