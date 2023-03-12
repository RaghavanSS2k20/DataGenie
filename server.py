from fastapi import FastAPI
from fastapi import FastAPI, Request, HTTPException
from fastapi import FastAPI, Request
from app import arima_model,xgpmodel
from ETS import forcast_ETS
app = FastAPI()

@app.get('/')
def index():
    return {"message":"app is ready"}

@app.post("/predict")

async def predict(date_from: str, date_to: str, period: int):
    body = await request.json()
    try:
        # Convert date strings to datetime objects
        date_from = datetime.strptime(date_from, '%Y-%m-%d')
        date_to = datetime.strptime(date_to, '%Y-%m-%d')
    except ValueError:
        raise HTTPException(status_code=400, detail='Invalid date format')
    
    # Ensure date_from is earlier than date_to
    if date_from > date_to:
        raise HTTPException(status_code=400, detail='Invalid date range')
    data = json.loads(body)
    df = pd.json_normalize(data)
    from helpers import helper_function
    test_data = helper_function(df)
    from main import gcls
    from app import fit
    pred= gcls.predict(test_data)

    predictions = []
    if pred == 0:
        testData = df['point_value']
        predictions=arima_model.predict(start=testData.index[0],end = testData.index[-1],typ = 'levels')
        mape = mean_absolute_percentage_error(test_data, predictions)
        predictions 

    elif pred == 1:
        predictions = fit.forecast(steps=period).tolist()
        mape = mean_absolute_percentage_error(df['point_values'], predictions)
        predictions = predictions.tolist()
    elif pred==2:
        testData = df
        df['point_timestamp']=pd.to_datetime(df['point_timestamp'])
        df['day'] = df['point_timestamp'].dt.day
        predictions=xgpmodel.predict(df['day'])
        mape = mean_absolute_percentage_error(df['point_values'], predictions)
        predictions = predictions.tolist()
    elif pred == 3:
        from app import prophet_model
        fdf = prophet_model.make_future_dataframe(periods=period,fred='D')
        pred = prophet_model.predict(fdf)
        predictions=pred['yhat'][-period:].values
        mape = mean_absolute_percentage_error(df['point_value'], predictions)
        predictions = predictions.tolist()
    modeldict = {0:'ARIMA',
            1:'ETS',
            2:'XGBOOST',
            3:'prophet'}
    response = {
        "Model":modeldict[pred],
        "mape":mape,
        "result":[
            {
                
            }
        ]
    }






    

    
        




    # Do prediction for given date range and period
    # prediction_result = your_model.predict(date_from, date_to, period)
    
    # Return prediction result
    return {'prediction': "hello" }
