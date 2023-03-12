import pandas as pd
import numpy as np
heads = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
heads.extend(['kurtosis','skew','stationary_value','sampling','slope','intercept','fft','mean_psd','std_psd','max_psd','max_freq','lowes_mape','model'])
mainDf = pd.DataFrame(columns=heads)
for i in range(1,11):
    df =  pd.read_csv("D:\DHACKATHON\DataGeinie\drive-download-20230310T132954Z-001\daily\sample_"+str(i)+".csv")
    df = df.dropna()
    from outliers import remove_outliers
    cdf = remove_outliers(df)

    from app import init
    # summary = cdf.describe()
    # print(summary)
    # heads = list(summary.index)
    print(heads)

    mainDf.columns = heads
    print(mainDf)
    lst = init(cdf)
    mainDf.loc[len(mainDf)] = lst
