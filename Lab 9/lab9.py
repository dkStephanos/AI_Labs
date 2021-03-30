import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for data visualizations
from sklearn.naive_bayes import GaussianNB
import seaborn as sns # for statistical data visualization
import warnings

warnings.filterwarnings('ignore')

data = 'adult.csv'
df = pd.read_csv(data, header=None, sep=',\s')

# STEP 7 ------------------------------------------------
col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship',
             'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']
df.columns = col_names


print(df.shape, df.head(), df.info())

# find categorical variables
categorical = [var for var in df.columns if df[var].dtype=='O']
print('There are {} categorical variables\n'.format(len(categorical)))
print('The categorical variables are :\n\n', categorical)

# view the categorical variables
print(df[categorical].head())

# check missing values in categorical variables
print(df[categorical].isnull().sum())

# view frequency counts of values in categorical variables
for var in categorical: 
    print(df[var].value_counts())

# view frequency distribution of categorical variables
for var in categorical: 
    print(df[var].value_counts()/np.float(len(df)))

# check labels in workclass variable
for var in ['workclass', 'occupation', 'native_country']:
    print(df[var].unique())

    # check frequency distribution of values in  variable
    print(df[var].value_counts())

    # replace '?' values in variable with `NaN`
    df[var].replace('?', np.NaN, inplace=True)

    # again check the frequency distribution of values in  variable
    print(df.workclass.value_counts())

print(df[categorical].isnull().sum())

# check for cardinality in categorical variables
for var in categorical:
    print(var, ' contains ', len(df[var].unique()), ' labels')

# find numerical variables
numerical = [var for var in df.columns if df[var].dtype!='O']
print('There are {} numerical variables\n'.format(len(numerical)))
print('The numerical variables are :', numerical)

# view the numerical variables
print(df[numerical].head())

# check missing values in numerical variables
print(df[numerical].isnull().sum())

# STEP 8 ------------------------------------------------
X = df.drop(['income'], axis=1)
y = df['income']

# STEP 9 ------------------------------------------------
# split X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# check the shape of X_train and X_test
print(X_train.shape, X_test.shape)

# STEP 10 ------------------------------------------------
# check data types in X_train
print(X_train.dtypes)

# display categorical variables
categorical = [col for col in X_train.columns if X_train[col].dtypes == 'O']
print(categorical)

# display numerical variables
numerical = [col for col in X_train.columns if X_train[col].dtypes != 'O']
print(numerical)

# print percentage of missing values in the categorical variables in training set
print(X_train[categorical].isnull().mean())

# print categorical variables with missing data
for col in categorical:
    if X_train[col].isnull().mean()>0:
        print(col, (X_train[col].isnull().mean()))

# impute missing categorical variables with most frequent value
for df2 in [X_train, X_test]:
    df2['workclass'].fillna(X_train['workclass'].mode()[0], inplace=True)
    df2['occupation'].fillna(X_train['occupation'].mode()[0], inplace=True)
    df2['native_country'].fillna(X_train['native_country'].mode()[0], inplace=True)    

# check missing values in categorical variables in X_train
print(X_train[categorical].isnull().sum())

# check missing values in categorical variables in X_test
print(X_test[categorical].isnull().sum())

# check missing values in X_train
print(X_train.isnull().sum())

# check missing values in X_test
print(X_test.isnull().sum())

# print categorical variables
print(categorical)
print(X_train[categorical].head())

# import category encoders
import category_encoders as ce

# encode remaining variables with one-hot encoding
encoder = ce.OneHotEncoder(cols=['workclass', 'education', 'marital_status', 'occupation', 'relationship', 
                                 'race', 'sex', 'native_country'])

X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)

print(X_train.head(), X_train.shape, X_test.head(), X_test.shape)

# STEP 11 ------------------------------------------------
cols = X_train.columns

from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = pd.DataFrame(X_train, columns=[cols])
X_test = pd.DataFrame(X_test, columns=[cols])
print(X_train.head())

# STEP 12 ------------------------------------------------

# train a Gaussian Naive Bayes classifier on the training set
from sklearn.naive_bayes import GaussianNB

# instantiate the model
gnb = GaussianNB()

# fit the model
gnb.fit(X_train, y_train)

# STEP 13 ------------------------------------------------
y_pred = gnb.predict(X_test)
print(y_pred)

# STEP 14 ------------------------------------------------
from sklearn.metrics import accuracy_score

print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

y_pred_train = gnb.predict(X_train)
print(y_pred_train)
print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))

# print the scores on training and test set

print('Training set score: {:.4f}'.format(gnb.score(X_train, y_train)))
print('Test set score: {:.4f}'.format(gnb.score(X_test, y_test)))

# check class distribution in test set
print(y_test.value_counts())

# check null accuracy score
null_accuracy = (7407/(7407+2362))
print('Null accuracy score: {0:0.4f}'. format(null_accuracy))

# STEP 15 ------------------------------------------------
# Print the Confusion Matrix and slice it into four pieces

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)

print('Confusion matrix\n\n', cm)
print('\nTrue Positives(TP) = ', cm[0,0])
print('\nTrue Negatives(TN) = ', cm[1,1])
print('\nFalse Positives(FP) = ', cm[0,1])
print('\nFalse Negatives(FN) = ', cm[1,0])

# visualize confusion matrix with seaborn heatmap

cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
plt.show()