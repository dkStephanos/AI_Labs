import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for data visualizations
from sklearn.naive_bayes import GaussianNB
import seaborn as sns # for statistical data visualization
import warnings

warnings.filterwarnings('ignore')

data = 'adult.csv'

df = pd.read_csv(data, header=None, sep=',\s')
col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship',
             'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']

df.columns = col_names

print(df.head())
