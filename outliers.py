import pandas as pd
import numpy as np
def remove_outliers(df):
    zscores = np.abs((df-df.mean())/df.std())
    threshold = 3
    outlier_indices = np.where(zscores > threshold)
    cleaned_df = df.copy()
    cleaned_df = cleaned_df.drop(index=outlier_indices[0])
    return cleaned_df

