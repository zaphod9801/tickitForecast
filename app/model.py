from tkinter import Y
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from math import sqrt
import itertools

def load_data():
    category_df = pd.read_csv("data/category_pipe.txt", sep="|", header=None, names=["catid", "catgroup", "catname", "catdesc"])
    event_df = pd.read_csv("data/allevents_pipe.txt", sep="|", header=None, names=["eventid", "venueid", "catid","dateid",  "eventname", "starttime"])
    venue_df = pd.read_csv("data/venue_pipe.txt", sep="|", header=None, names=["venueid", "venuename", "venuecity", "venuestate", "venueseats"])
    listing_df = pd.read_csv("data/listings_pipe.txt", sep="|", header=None, names=["listid", "sellerid", "eventid", "dateid", "numtickets", "priceperticket", "totalprice", "listtime"])
    date_df = pd.read_csv("data/date2008_pipe.txt", sep="|", header=None, names=["dateid", "caldate", "day", "week", "month", "qtr", "year", "holiday"])
    users_df = pd.read_csv("data/allusers_pipe.txt", sep="|", header=None, names=["userid", "username", "firstname", "lastname", "city", "state", "email", "phone", "likesports", "liketheatre", "likeconcerts", "likejazz", "likeclassical", "likeopera", "likerock", "likevegas", "likebroadway", "likemusicals"])
    sales_df = pd.read_csv("data/sales_tab.txt", sep="\t", header=None, names=["salesid", "listid", "sellerid", "buyerid", "eventid", "dateid", "qtysold", "pricepaid", "commission", "saletime"])
    
    # Combinar las tablas
    sales_event_df = pd.merge(sales_df, event_df, on="eventid")
    sales_event_venue_df = pd.merge(sales_event_df, venue_df, on="venueid")
    sales_event_venue_date_df = pd.merge(sales_event_venue_df, date_df, left_on="dateid_x", right_on="dateid", suffixes=("", "_date"))
    sales_event_venue_date_category_df = pd.merge(sales_event_venue_date_df, category_df, on="catid")
    
    # Procesar la columna 'saletime' para extraer la fecha sin la hora
    sales_event_venue_date_category_df["sale_date"] = pd.to_datetime(sales_event_venue_date_category_df["saletime"]).dt.date
    
    return sales_event_venue_date_category_df
    
def split_data_sarimax():
    sales_event_venue_date_category_df = load_data()
    # Agrupar por 'sale_date' y sumar la cantidad vendida (qtysold)
    daily_sales_df = sales_event_venue_date_category_df.groupby("sale_date").agg({"qtysold": "sum"}).reset_index()
    
    # Ordenar por fecha
    daily_sales_df.sort_values("sale_date", inplace=True)
    
    # Extraiga las características (X) y la variable objetivo (y)
    X = daily_sales_df["sale_date"].values.reshape(-1, 1)
    y = daily_sales_df["qtysold"].values
        
    # Divida los datos en conjuntos de entrenamiento (80%) y prueba (20%) según la fecha
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return y_train, y_test


def grid_search(train, test, p_values, d_values, q_values, P_values, D_values, Q_values, s_values):
    best_score, best_params, best_seasonal_params = float("inf"), None, None
    best_model, best_preds = None, None
    
    # Crea una lista de combinaciones de parámetros
    pdq = list(itertools.product(p_values, d_values, q_values))
    seasonal_pdq = list(itertools.product(P_values, D_values, Q_values, s_values))

    for order in pdq:
        for seasonal_order in seasonal_pdq:
            try:
                model = SARIMAX(train, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
                model_fit = model.fit(disp=0)
                preds = model_fit.forecast(steps=7)
                rmse = mean_squared_error(test[:7], preds, squared=False)

                if rmse < best_score:
                    best_score, best_params, best_seasonal_params = rmse, order, seasonal_order
                    best_model, best_preds = model_fit, preds

            except Exception as e:
                print(f"Error with parameters: {order}, {seasonal_order}")
                print(e)
                continue

    return best_model, best_preds, best_score, best_params, best_seasonal_params


    
def prediction_sarimax():
    y_train, y_test = split_data_sarimax()

    print(y_train)
    p_values = range(0, 2)
    d_values = range(0, 2)
    q_values = range(0, 2)
    P_values = range(0, 2)
    D_values = range(0, 2)
    Q_values = range(0, 2)
    s_values = [7]  # Semanal

    best_model, best_preds, best_score, best_params, best_seasonal_params = grid_search(y_train, y_test, p_values, d_values, q_values, P_values, D_values, Q_values, s_values)

    print(f"Best parameters: {best_params}, Best seasonal parameters: {best_seasonal_params}")
    print(f"RMSE: {best_score}")

    return best_preds
    
def aggregate_data_by_date(data):
    aggregated_data = data.groupby("sale_date")["qtysold"].sum().reset_index()
    return aggregated_data

def create_lag_features(data, n_lags):
    df = data.copy()
    for i in range(1, n_lags + 1):
        df[f"lag_{i}"] = df["qtysold"].shift(i)
    df.dropna(inplace=True)
    return df

def train_linear_regression(data, n_lags):
    lagged_data = create_lag_features(data, n_lags)
    X = lagged_data.drop(columns=["qtysold", "sale_date"]).values
    y = lagged_data["qtysold"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    return model


def prediction_linear_regression():
    data = load_data()
    aggregated_data = aggregate_data_by_date(data)
    n_lags = 7
    model = train_linear_regression(aggregated_data, n_lags)
    
    # Make predictions for the next 7 days
    last_n_days = create_lag_features(aggregated_data, n_lags).tail(1).drop(columns=["qtysold", "sale_date"]).values
    predictions = model.predict(np.repeat(last_n_days, 7, axis=0))
    
    return predictions


if __name__ == '__main__':
    print(prediction_sarimax())
    #prediction_linear_regression()