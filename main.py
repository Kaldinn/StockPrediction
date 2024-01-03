import yfinance as yf
from sklearn.ensemble import RandomForestClassifier


sp500 = yf.Ticker("^GSPC")
sp500 = sp500.history(period="max")

model = RandomForestClassifier(n_estimators=100,min_samples_split=100, random_state=1)