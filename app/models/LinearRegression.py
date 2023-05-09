import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from app.data import load_data, split_data_linear_regression, aggregate_data_by_date, create_lag_features



def train_linear_regression(data, n_lags):
    X_train, X_test, y_train, y_test = split_data_linear_regression(data, n_lags)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    return model, mse


def prediction_linear_regression():
    data = load_data()
    aggregated_data = aggregate_data_by_date(data)
    n_lags = 7
    model, mse = train_linear_regression(aggregated_data, n_lags)
    
    # Make predictions for the next 7 days
    last_n_days = create_lag_features(aggregated_data, n_lags).tail(1).drop(columns=["qtysold", "sale_date"]).values
    predictions = model.predict(np.repeat(last_n_days, 7, axis=0))
    
    return predictions, mse

#print(prediction_linear_regression())