import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression

def train_model(df):

    df["rain_label"] = df["precipitation_sum"].apply(lambda x: 1 if x > 0 else 0)

    X = df[[
        "temperature_2m_max",
        "temperature_2m_min",
        "wind_speed_10m_max"
    ]]

    y = df["rain_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression()

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    report = classification_report(y_test, predictions)

    return model, accuracy, report


def predict_rain(model, data_row):

    data_row = [float(x) for x in data_row]

    prob = model.predict_proba([data_row])[0][1]

    prediction = model.predict([data_row])[0]

    return prediction, prob