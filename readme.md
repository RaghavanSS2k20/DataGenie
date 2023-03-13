# DataGenie Hackathon - 2023
## _prediction of best time series model_
idea is that each time series data has its own metrics, like mean, std, kurtosis, and has its own sampling. So by analysing it and retrieving all the metrics, we find best model for that timeseries data and store all the metrics of the timeseries data as X and best model as Y so by using this as a dataset Gaussian Naive bayes model is trained 



## Concepts used

#### **->** Python
#### **->** Scikit
#### **->** Gaussian Naive bayes classifier
#### **->** FastAPI
#### **->** Flask app for routing the frontend

# Features
### kurtosis: 
    Kurtosis is a measure of the "tailedness" of a distribution. It measures the extent to which the tails of a distribution differ from those of a normal distribution. A kurtosis value of 0 indicates a normal distribution, positive values indicate heavy tails, and negative values indicate light tails.

### skew: 
    Skewness is a measure of the asymmetry of a distribution. It measures the extent to which a distribution deviates from symmetry around its mean. A skewness value of 0 indicates a perfectly symmetric distribution, positive values indicate a right-skewed distribution, and negative values indicate a left-skewed distribution.

### stationary_value: 
   mathematical value for analysing stationarity in data.

### sampling: 
    Sampling refers to the process of selecting a subset of observations from a larger population. In the context of time series analysis, sampling refers to the process of selecting a subset of time points for analysis.

### slope: 
    Slope refers to the steepness of a line. In the context of time series analysis, slope of the linear regression model for the trend analysis.

### intercept: 
    Intercept refers to the value of the dependent variable when the independent variable is zero.
    In the context of time series analysis, intercept of the linear regression model for the trend analysis..
### The slope and intercept of linear regression of time series data can help in trend analysis by providing information about the direction and magnitude of the trend.

### fft: 
    FFT stands for Fast Fourier Transform, which is a mathematical algorithm used to transform time-domain data into frequency-domain data. By analyzing the frequency components of the data using FFT, we can identify the presence of any periodic patterns or cyclical behavior that may indicate the presence of a trend in the data..

### mean_psd: 
    Mean power spectral density (PSD) is a measure of the average power of a signal over its entire frequency range.

### std_psd: 
    Standard deviation of power spectral density (PSD) is a measure of the variability of the power of a signal over its entire frequency range.

### max_psd: 
    Maximum power spectral density (PSD) is the highest power value observed in the frequency spectrum of a signal.

### max_freq: 
    Frequency at which the maximum power spectral density (PSD) occurs.


### all mean_psd, std_psd, max_psd, max_freq can be used to tell about stationarity of the time series data
### model:
    Best time series model for the given dataset


## API endpoints
### /predict?date_from=<from date>&date_to=<to date>&period=0
```
for index,row in df.iterrows():
        result.append({'point_timestamp':row['point_timestamp'],
                        'point_value':row['point_value'],
                        'yhat':predictions[index]})
response = {
        "Model":modeldict[pred],
        "mape":mape,
        "result":result
    }
```

### /plot/roc
```
    response={
                'fpr':fpr.tolist(),
                'tpr':tpr.tolist()
            }
            
    """ js file fetches this data and plots roc curve"""
    
    
    <script>
        fetch('http://localhost:8000/plot/roc')
          .then(response => response.json())
          .then(data => {
            // Use data to plot chart using Chart.js
            var ctx = document.getElementById('myChart').getContext('2d');
            var chart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: data.fpr,
                    datasets: [{
                        label: 'ROC curve',
                        borderColor: 'blue',
                        data: data.tpr,
                        fill: false
                    }]
                },
                options: {
                    scales: {
                        xAxes: [{
                            type: 'linear',
                            position: 'bottom',
                            ticks: {
                                max: 1,
                                min: 0
                            }
                        }]
                    }
                }
            });
          });
    </script>




### 20PW26 - RAGHAVAN M
