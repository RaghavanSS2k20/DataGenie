from statsmodels.tsa.exponential_smoothing.ets import ETSModel
def forcast_ETS(df):
    try:
        
        model = ETSModel(df,error='add', trend='add', seasonal='add', seasonal_periods=365)
        fit = model.fit()
        forecast = fit.forecast(steps=365)
        return forecast
    
    except ValueError:
        return "THIS MODEL CANNOT BE USED FOR THIS DATA"

