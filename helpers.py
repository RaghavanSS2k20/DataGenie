import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.fft import fft
def helper_function(cleaned_df):
    adfTestResult = adfuller(cleaned_df['point_value'])
    f,psd = signal.welch(cleaned_df['point_value'], fs=1, nperseg=256)
    mean_psd = np.mean(psd)
    std_psd = np.std(psd)
    max_psd = np.max(psd)
    max_freq = f[np.argmax(psd)]
    x = cleaned_df.iloc[:,1].values.astype(float)
    y = cleaned_df.iloc[:,2].values
    coefficients = np.polyfit(x, y, deg=1)
    from scipy.fft import fft
    fftval = fft(cleaned_df['point_value'].values)
    dc_component = np.real(fftval[0]) / len(cleaned_df['point_value'])
    summary = cleaned_df.describe()
    l = [summary.loc['count', 'point_value'], summary.loc['mean', 'point_value'],summary.loc['std', 'point_value'],summary.loc['min', 'point_value'],summary.loc['max', 'point_value'],summary.loc['25%', 'point_value'],summary.loc['50%', 'point_value'],summary.loc['75%', 'point_value'],
    cleaned_df['point_value'].kurtosis(),cleaned_df['point_value'].skew(),adfTestResult[1],f,coefficients[0],coefficients[1],dc_component,mean_psd,std_psd,max_psd,max_freq,min(mapes)]
    return l
