import pandas as pd
from sklearn.metrics import precision_score

def backtest(model, sp500, predictors):
    test = sp500.iloc[-100:]
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index)
    precision = precision_score(test["Target"], preds)

    combined = pd.concat([test["Target"], preds], axis=1)
    combined.plot()

    return combined, precision

def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:,1]
    preds[preds >= .6] = 1
    preds[preds < .6] = 0
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

