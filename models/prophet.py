import pandas as pd
from prophet import Prophet
from sklearn.model_selection import train_test_split

def prophet_predict(train_data, test_data,f):
    model = Prophet()
    print('prrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr',f)
    print(train_data)
    
    train_data = train_data.iloc[:,[1,2]]
    test_data = test_data.iloc[:,[1,2]]
    train_data.columns = ['ds', 'y']
    test_data.columns = ['ds', 'y']
    model.fit(train_data)
    
    
    fdf = model.make_future_dataframe(periods = len(test_data),freq=f)
    pred = model.predict(fdf)
    return [test_data['y'].values,pred['yhat'][-len(test_data):].values,model]
