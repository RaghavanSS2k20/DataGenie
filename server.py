from fastapi import FastAPI
from fastapi import FastAPI, Request, HTTPException
from fastapi import FastAPI, Request
from app import arima_model
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
    pred= gcls.predict(test_data)
    predictions = []
    if pred == 0:
        testData = df['point_value']
        predictions=arima_model.predict(start=testData.index[0],end = testData.index[-1],typ = 'levels').tolist()
        mape = mean_absolute_percentage_error(test_data, predictions)

    if pred == 1:
        


    # Do prediction for given date range and period
    # prediction_result = your_model.predict(date_from, date_to, period)
    
    # Return prediction result
    return {'prediction': "hello" }
