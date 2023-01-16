import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing, model_selection
import pandas as pd

style.use('ggplot')

# df = pd.read_excel('titanic.xls')
# df.to_csv("titanic.csv")
df = pd.read_csv("titanic.csv")
df.drop(['body', 'name'], 1, inplace=True)
df = df.apply(pd.to_numeric, errors='ignore')
df.fillna(0, inplace=True)
# print(df.head())

def handle_non_numeric_data(df):
    columns = df.columns.values
    for col in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]
        if df[col].dtype != np.int64 and df[col].dtype != np.float64:
            column_contents = df[col].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x += 1
            df[col] = list(map(convert_to_int, df[col]))
    return df

df = handle_non_numeric_data(df)
# print(df.info())

