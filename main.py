import pandas as pd
import numpy as np
heads = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
heads.extend(['kurtosis','skew','stationary_value','sampling','slope','intercept','fft','mean_psd','std_psd','max_psd','max_freq','model'])
mainDf = pd.DataFrame(columns=heads)
def checkcollinear(X,Y,THRESHOLD):
    corr = np.cov(X,Y)[0][1]/(np.std(X)*np.std(Y))
    if (corr >= THRESHOLD or corr <= -THRESHOLD):
        return True
    else:
        return False
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
    lst = init(cdf,'D')
    mainDf.loc[len(mainDf)] = lst
for i in range(1,11):
    df =  pd.read_csv("D:\DHACKATHON\DataGeinie\drive-download-20230310T132954Z-001\hourly\sample_"+str(i)+".csv")
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
    lst = init(cdf,'H')
    mainDf.loc[len(mainDf)] = lst
for i in range(0,10):
    df =  pd.read_csv("D:\DHACKATHON\DataGeinie\drive-download-20230310T132954Z-001\monthly\sample_"+str(i)+".csv")
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
    lst = init(cdf,'M')
    mainDf.loc[len(mainDf)] = lst
for i in range(1,8):
    df =  pd.read_csv("D:\DHACKATHON\DataGeinie\drive-download-20230310T132954Z-001\weekly\sample_"+str(i)+".csv")
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
    lst = init(cdf,'W')
    mainDf.loc[len(mainDf)] = lst


from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = mainDf
data.replace("D",0)
data.replace("H",1)
data.replace("M",2)
data.replace("W",3)


data.replace('ARIMA',0)
data.replace('ETS',1)
data.replace("XGBOOST",2)
data.replace("Prophet",3)
THRESHOLD = 0.7
data = data.apply(pd.to_numeric)
features = list(data.head())
corr = data.corr()
combination = combinations(features[:-1],2)
for i in list(combination):
    if(checkcollinear(data[i[0]], data[i[1]], THRESHOLD) and i[0] in features and i[1] in features):
        features.remove(i[1])
data = data[features]
X_train, X_test, y_train, y_test = train_test_split(data.drop('model',axis=1), data['model'], test_size=0.3, random_state=42)
classifier = GaussianNB()
classifier.fit(X_train,y_train)
pred = classifier.predict(X_test)
gcls = classifier
accuracy = accuracy_score(y_true, y_pred)
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, roc_auc_score
report = classification_report(y_test, pred)
print(report)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)