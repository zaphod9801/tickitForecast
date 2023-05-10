from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import itertools
from app.data import split_data_sarimax
    


def grid_search(train, test, p_values, d_values, q_values, P_values, D_values, Q_values, s_values):
    best_score, best_params, best_seasonal_params = float("inf"), None, None
    best_model, best_preds = None, None
    
    # Crear una lista de combinaciones de parámetros
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

    return best_preds, best_score, best_params
    
#print(prediction_sarimax())