from fastapi import FastAPI
from fastapi import FastAPI, Request, HTTPException
from fastapi import FastAPI, Request
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
    
    # Do prediction for given date range and period
    # prediction_result = your_model.predict(date_from, date_to, period)
    
    # Return prediction result
    return {'prediction': 'your_prediction_result'}
