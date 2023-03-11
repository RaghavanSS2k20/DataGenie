import xgboost as xgb

def xg_prediction(train_data,test_data):
    X_train = train_data.drop('point_value', axis=1)
    y_train = train_data['point_value']
    X_test = test_data.drop('point_value', axis=1)
    y_test = test_data['point_value']
    xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)
    y_pred = xgb_model.predict(X_test)
    return [y_pred,y_test]
