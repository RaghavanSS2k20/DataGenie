import pandas as pd
import numpy as np
<<<<<<< HEAD
import sys
=======
>>>>>>> parent of 364441e6 (xgboost and ets)

sys.path.append('D:/XB')
from test import xgboost_predict
from scipy.stats import norm
from tsfresh import *
from tsfresh.utilities.dataframe_functions import impute
from multiprocessing import Pool, freeze_support
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller
from scipy import signal
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error
# from pmdarima.arima import auto_arima

from statsmodels.tsa.stattools import adfuller
if __name__ == '__main__':
    df =  pd.read_csv("D:\DHACKATHON\DataGeinie\drive-download-20230310T132954Z-001\daily\sample_1.csv")
    print(type(df['point_value'][1]))
    df = df.dropna()
    # plt.plot(df['point_timestamp'],df['point_value'])


    # plt.title('Title of the Plot')
    # plt.xlabel('Date')
    # plt.ylabel('point_value')
    # plt.show()
    # plt.close()


    # plt.plot(df['point_timestamp'],df['point_value'])
    # mu = np.mean(df['point_value'])
    # sigma = np.std(df['point_value'])
    # x = np.linspace(df['point_value'].min(), df['point_value'].max(), 100)
    # y = norm.pdf(x, mu, sigma)
    # plt.plot(x, y, color='r')
    # plt.title('Normal Distribution of Time Series Data')
    # plt.xlabel('Value')
    # plt.ylabel('Density')
    # plt.show()


    # rolling_mean = df['point_value'].rolling(window=30).mean()
    # rolling_std = df['point_value'].rolling(window=30).std()

    # # Plot the time series along with its rolling mean and rolling standard deviation
    # plt.plot(df['point_value'], color='blue', label='Original')
    # plt.plot(rolling_mean, color='red', label='Rolling Mean')
    # plt.plot(rolling_std, color='black', label='Rolling Std')
    # plt.legend()
    # plt.title('Rolling Mean and Standard Deviation')
    # plt.show()

    # Perform the ADF test and print the p-value
    # result = adfuller(df['point_value'])
    # print('ADF p-value:', result[1])

    plot_acf(df['point_value'])
    plt.show()

    from outliers import remove_outliers
    df.dropna()
    cleaned_df = remove_outliers(df)
    print(df)
    print(cleaned_df)
    cleaned_df.hist()
    plt.show()

    adfTestResult = adfuller(cleaned_df['point_value'])
    print(adfTestResult[1])
    
    f=extract_features(cleaned_df,column_id='point_value',column_sort='point_timestamp' )
    f.to_csv('test3.csv')
    # f = tsfresh.feature_extraction.feature_calculators.abs_energy(df['point_timestamp'])
    agg = f.agg(['mean'])
    # feature_vector = np.concatenate(agg)
    # print(feature_vector)
    agg.to_csv("test.csv")
    summary = cleaned_df.describe()
    mainDf = pd.DataFrame()
    heads = list(summary.index)
    heads.extend(['kurtosis','skew','stationary_value','sampling','autocorrelation','trend_x','trend_y','mean_psd','std_psd','max_psd','max_freq','lowes_mape','model'])
        
    f,psd = signal.welch(cleaned_df['point_value'], fs=1, nperseg=256)
    mean_psd = np.mean(psd)
    std_psd = np.std(psd)
    max_psd = np.max(psd)
    max_freq = f[np.argmax(psd)]
    mainDf.head=heads
    
    mapes = []
    ###             GETTING MAPE VALUES
    ## ARIMA MODEL
    from statsmodels.tsa.arima.model import ARIMA
    from sklearn.metrics import mean_absolute_percentage_error
    from sklearn.model_selection import train_test_split
    cleaned_df['point_timestamp'] = pd.to_datetime(cleaned_df['point_timestamp'])
    cleaned_df.drop(['Unnamed: 0'],axis=1)
    
    
    train_data, test_data = train_test_split(cleaned_df.iloc[:,2], test_size=0.2, shuffle=False)
    print(train_data)
    arima_model = ARIMA(endog=train_data, order=(2, 1, 0)).fit()
    print("test DATA : ")
    print(test_data)
    predictions = arima_model.predict(start=test_data.index[0], end=test_data.index[-1], typ='levels')
    mape = mean_absolute_percentage_error(test_data, predictions)
    print(predictions)

    print(mape)
    mapes.append(mape)


    ##Prophet model
    
    
    





    

