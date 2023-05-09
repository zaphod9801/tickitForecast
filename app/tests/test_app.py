from app.models.Sarimax import prediction_sarimax
from app.models.LinearRegression import prediction_linear_regression

def test_prediction_sarimax():
    best_preds, best_score, best_params = prediction_sarimax()
    assert len(best_preds) > 0
    assert best_score != None
    assert len(best_params) > 0
    
def test_prediction_linear():
    prediction, mse = prediction_linear_regression()
    assert len(prediction) > 0
    assert mse != None

    
