import pandas as pd
import numpy as np

from scipy.stats import norm
from tsfresh import *
from tsfresh.utilities.dataframe_functions import impute
from multiprocessing import Pool, freeze_support
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

from statsmodels.tsa.stattools import adfuller
if __name__ == '__main__':
    df =  pd.read_csv("D:\DATAFIENE\datagenie-hackathon\drive-download-20230310T132954Z-001\daily\sample_1.csv")
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
    print("                   ")
    
    f=extract_features(df,column_id='point_value',column_sort='point_timestamp' )
    f.to_csv('test2.csv')
    # f = tsfresh.feature_extraction.feature_calculators.abs_energy(df['point_timestamp'])
    agg = f.agg(['mean', 'std', 'min', 'max'])
    # feature_vector = np.concatenate(agg)
    # print(feature_vector)
    agg.to_csv("test.csv")


    

