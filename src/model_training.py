from sklearn.ensemble import RandomForestClassifier

def train_model(sp500, predictors):
    model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)
    train = sp500.iloc[:-100]
    model.fit(train[predictors], train["Target"])
    return model
