from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
import optuna
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.decomposition import KernelPCA  # Kernel PCA
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer
from glob import glob
from os import path
import re
from IPython.core.display import display, HTML
import matplotlib as mpl
import psycopg2
import pickle
import pymssql
from sqlalchemy import create_engine
import urllib.parse
from fuzzywuzzy import process
import fuzzywuzzy
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from category_encoders import MEstimateEncoder
from sklearn.decomposition import PCA
from yellowbrick.target import FeatureCorrelation
from sklearn import datasets
from sklearn.feature_selection import mutual_info_regression
from datetime import datetime
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from category_encoders import BinaryEncoder
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
import warnings
import matplotlib.pyplot as plt  # from matplotlib import pyplot as plt
import seaborn as sns
from pyrsistent import b
import pandas as pd
%load_ext autotime  # At the beginning of the jupyter notebook to measure the execution time
%unload_ext autotime  # To unload
# Settings up the black (python formatter in VScode)
https: // dev.to/adamlombard/how-to-use-the-black-python-code-formatter-in-vscode-3lo0
https: // flake8.pycqa.org/en/latest/
https: // pre-commit.com/

# change the width of the jupyter notebook
%pylab inline
display(HTML("<style>.container { width:90% !important; }</style>"))
# %matplotlib inline
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)  # Set columns mac without truncated
pd.set_option('display.max_rows', None)  # Set display max

with pd.option_context('display.max_colwidth', None):  # showing max column
  display(df)

df[~df['Hotel'].str.contains('CZK|Hyatt.com|Trip.com|Hotels.com|Sponsored|Booking.com|Agoda.com|Expedia.com|FindHotel').fillna(
    False)].reset_index(drop=True)

df = pd.read_csv('https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv',
                 usecols=['hotel', 'is_canceled', 'adults', 'children', 'babies', 'meal', 'country', 'market_segment', 'distribution_channel', 'reserved_room_type', 'booking_changes', 'deposit_type', 'days_in_waiting_list', 'customer_type', 'required_car_parking_spaces', 'total_of_special_requests'])


def find_ext(dr, ext):
    return glob(path.join(dr, f"*.{ext}"))
# glob to search the same file in the directory


find_ext(".", "pdf")

for i in glob("*.pdf"):
    print(i)

# set for large data or assign data type to the particular columns
df.pd.read_csv('file_name.csv', low_memory=False)

df = pd.concat([df_1, df_2], axis=0)  # concat both dataframe with same columns
dfa.combine_first(dfb)  # fastest way to concat two dataframe
pd.concat([dfa, dfb])  # or this one

df.sort_values("column_name")

df.info()  # Basic info of the table such as type
df.info(memory_usage='deep')  # The memory usage
df.memory_usage(deep=True)  # breakdown of the memory usage each columns

# Distribution data
df.describe()
df['column_name'].describe()
df['column_name'].quantile(0.1)  # Find the quantile for the distribution
df['column_name'].hist()  # Plot the histogram and distribution
df['column_name'].plot(kind='kde')

df.drop_duplicates(keep=False)  # drop duplicates row in the dataframe
# drop duplicates based on the column_name
df.drop_duplicates(subset=['column_name'], keep=False)

df.isnull().sum()  # Row with NaN value

df.isnull().sum().sum()  # Total NaN value

df.fillna(0, inplace=True)  # fill NaN value with 0

# find max value in row
df['max_value'] = df.max(axis=1)
df[['a', 'b', 'c']].idxmax(1)  # return which column

df.loc[:, df_error_code].sum(axis=1)

# count the length of each row in a column and get the row which length row more than 0
df[df['column1'].str.len() > 0]

df[df.isnull().T.any()]  # || df[df.isnull().any(axis=1)] #Rows with NaN value

df['column_1'] = df['column_1'].replace(
    [332, 333, 342], 6)  # replace 332,333,342 with 6

df.melt(id_vars=['column_name'])

pd.pivot_table(df, values='column1', index=['column2', 'column3'], columns=['column4'],
 aggfunc={'column5': np.mean, 'column6': [min, max, np.mean]}, fill_value=0)  # pivot and sum

df['hotel'].value_counts().plot.bar()  # count the main column value and plot

list(df), df.columns.tolist()  # List all long columns names
df.index.values.tolist()  # get the index of the df and put into a list

# stack dataframes with similar columns, ignore index
df1.append(df2, ignore_index=True)

pd.crosstab(df['hotel'], df['is_canceled'], margins=True, margins_name='Total')
pd.crosstab(df['adults'], df['children'], margins=True, margins_name='Total')

df1.set_index('column1').combine_first(df2.set_index('column1')
              ).reset_index()  # combine two uneven df columns

df.isna().sum()/len(df)*100  # Percentage of NaN value

# groupby daily mean of column1
df.resample('D', on="datetime")['column1'].mean()

# spread the string column by '-' and select the second column
df['column_1'].str.split('-', expand=True)[1].reset_index(drop=True)
df['column_name'].str.split('/', expand=True)[[0, 1]].reset_index(drop=True).rename(
    columns={0: 'continent', 1: 'city'})  # spread and rename two columns
# remove the column and join, get only the first split and rename the columns
df.join(df.pop('column_name').str.split('-', n=1, expand=True)
        ).rename({0: 'fips', 2: 'row'}, axis=1)
# get the first value after the split and split by comma(,)
df['column_name'].str.split(",").str.get(0)
# split the second element and leave the first
df.index.str.rsplit("-",  n=2, expand=True)
# Difference between split and rsplit are rsplit has n parameters and split from last element

df.dtypes  # Get the type of the column data
# Get the list of object category
list(df.select_dtypes(include=['object']).columns)

df['Target'] = np.where(df['column_name'].isin(
    df1['column_name']), 'Fail', 'Pass')  # create failed pass column

df.drop(['columns2', 'column1'], axis=1)  # Drop columns
list_aw = ["a", "b"]
df.drop(df[df['Filename'].isin(list_aw)].index)  # drop row based on filename

# iterrows access row value
for index, row in df.iterrows():  # dataframe
    print(row['column_name'])
    print(index)

# iterrows append new dataframe
new_df = pd.DataFrame()
for index, row in df.iterrows():
    new_df = new_df.append(row)

# sample namedtuple
Point = namedtuple('Point', ['x', 'y'])
points = [Point(1, 2), Point(3, 4)]
pd.DataFrame(points, columns=Point._fields)

# regex findall
text = "string"  # some string
re.findall(
    "[a-zA-Z0-9]{1,10}-[a-zA-Z0-9]{1,10}-[a-zA-Z0-9]{1,10}-[a-zA-Z0-9]{1,10}G", text)  # ['80-56-17801-512G', '80-56-17801-512G']
re.findall("[0-9]{10}", text)  # ['1965917932']
# ['130MB/s', '130MB/s', '130MB/s', '130MB/s', '130MB/s']
re.findall("[0-9]{1,10}MB/s", text)
re.findall("Made in [a-zA-Z1-9]{1,10}", text)  # ['Made in China']


# iterate dataframe and append to new dataframe using itertuples
temp_df = pd.DataFrame()
for row in df.itertuples(index=False):
    temp_df = temp_df.append(pd.DataFrame([row], columns=row._fields))
temp_df.reset_index(drop=True)

for index, value in df['column_name'].iteritems():  # dataseries
    print(index)
    print(value)

s = pd.Series(['A', 'B', 'C'])
for index, value in s.items():
    print(f"Index : {index}, Value : {value}")

df.groupby([pd.Grouper(freq='5min', key='date_column'), 'column1',
           'column2']).size().unstack(fill_value=0)  # Group data by 5 min
# https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases

# groupby column2 and find unique record for column2 and find value more than 1
df.groupby(['column1'])['column2'].nunique().reset_index(
    name="count").query('count > 1')
# will list out the unique value in column2
df.groupby(['column1'])['column2'].unique()

# Assign two variables with same value
variable_1 = variable_2 = df['datetime_column'].min()

# count occurances in a list
a = ['a', 'b', 'r', 'a', 'c', 'a', 'd', 'a', 'b', 'r', 'a']
Counter(a).most_common()

# compare two lists if both exists in those two lists
np.intersect1d(list_1, df.columns).tolist()
# compare two lists and show only the difference
np.setdiff1d(list_1, df.columns).tolist()

# get the time for last 5 minutes
five_mins_ago = datetime.datetime.now() - datetime.timedelta(minutes=5)
d = datetime.today() - timedelta(hours=0, minutes=50)  # hours and minutes

# Query
df.query('column_name == "string_value"')
df.query("column_name.str.startswith('D') or column_name.str.startswith('C')",
         engine='python').reset_index()
df.query("column == 'something' & column == 'something' | column == ''")  # emptynan
# unique value for each columns
df.query("column == 'something' & column == 'something' | column == ''").nunique()
df.query('column1 == "something" & column2 <= "something2" & (column3 == 0 | column4 == 6)').nunique()  # or and &

# Checking duplicate column or checking the value in both columns are the same
df.T.drop_duplicates().T
df[df['column_same1'] != df['column_same2']]  # checking the diff of the column

# Apply groupby
g[['column1', 'column2']].apply(lambda x: x / x.sum())

# change column type and add zfill
df['column1'].astype(str) + '-' + df['column2'].astype(str).str.zfill(2) + '-' + \
                     df['column3'].astype(str).str.zfill(
                         2) + '-' + df['column4'].astype(str).str.zfill(2)

df['column_int'] = df['column_int'].astype(
    int)  # change the column type to int

df.loc[(df['column_name'] == 'somev_alue') | (
    df['column2'] >= 90)]  # alternative to query

idx = df['column1'].eq('a').idxmax()  # find the a in column1
df.loc[[idx]]

# get dataframe last row of containing a
idx = ((df['column1'] != 'a').idxmax() - 1)
df.loc[0:idx]

# swap the position of dataframe bottom to up and get the dataframe for only contains a
idx = ((df[::-1]['column1'] != 'a').idxmax() + 1)
df[::-1].loc[:idx]

pd.merge(df1, df2, on="columns")
pd.merge(df1, df2, how="inner", validate='one_to_many')

# code below produce the same output with concat will not rename the duplicate columns and merge will rename it
pd.merge(df1, df2.drop(['columnToDrop'], axis=1),
         left_index=True, right_index=True, how='left')
pd.concat([df1, df2.drop(['columnToDrop'], axis=1)], axis=1)

df['column_name'].min() | df['column_name'].max()

pd.to_datetime(df['column_name'], format="%m/%d/%y")
df[df['date_time_column'].between(
    "2022-10-01 00:20:00|start_time", "2022-10-01 00:40:00|end_time")]
df.between_time('0:15', '0:45')  # time between 15 minutes to 45 minutes

plt.figure(figsize=(15, 8))  # Plot the data with threshold
df_count_country = df[3].value_counts()
df_count_country['other'] = df_count_country[df_count_country < 10].sum()
df_count_country = df_count_country[df_count_country > 10]
df_count_country.plot(kind='bar')

# remove UTC from the string in pandas column
df['column'].str.replace('UTC ', '')

df.groupby(['column 1', 'column 2']).size(
).reset_index(name='counts')  # Groupby

df[df.duplicated()]  # check for duplicate in any column
# Check duplicate value for a particular column
df[df.duplicated(subset=['column_name'], keep=False)]

pd.to_datetime(df['Column'])  # change column to datetime

datetime.today().strftime('%Y-%m-%d-%H:%M:%S')  # Get the date for today

# from datetime import datetime as dt
# import pytz
dt.now().astimezone(pytz.timezone('Asia/Kuala_Lumpur')
       ).strftime("%a %d-%m-%Y  %H:%M:%S")

df.T  # transpose
df.describe().T  # transpose describe

df['spec_version'].value_counts().reset_index().rename(
    columns={'index': 'spec_version', 'spec_version': 'count'})
# checking if the columns exist and select only the existing columns
df[df.columns.intersection(set(['0006', '0332', '0333', '0342']))]
df[df['Column name'].notnull()]  # find row with not nan value
df[df['Column name'].isnull()]  # find row with nan value

df.columns = ['Column 1', 'Column 2', 'Column 3']  # rename all columns

[col[0] for col in df.columns]  # multiindex column

# x is the value of the row in the column_1
df['column_1'].apply(lambda x: 'PRMNG_' + x if x == '0006' else 'FTNG_' + x)

df.to_csv('df_name.csv', encoding='utf-8')  # save to csv
path = '/home/path/work/path/Data/'
df.to_csv(path+'df_name.csv', index=False)

# Run the processing using the categorical preprocessing
cat_columns = list(df.select_dtypes(include=['object']).columns)
df[cat_columns] = df[cat_columns].apply(
    lambda x: x.astype('category').cat.codes)

df[~df['column_name'].isin(list_of_data)]  # isin

df.rename(columns={"Destination": "iata_code"},
          inplace=True)  # rename the column

df.add_suffix('_some_suffix')  # rename all columns with suffix
df.add_prefix('some_prefix_')  # rename all columns with prefix
df.rename('some_prefix_{}'.format, axis=1)

# showing Column_name3 and groupby by column_name1 & column_name2
df.groupby(['Column_name1', 'Column_name2'])['Column_name3'].sum()

df1 = df.groupby(['column_name1', 'column_name2'], as_index=False).agg(
    {'column_datetime': ['min', 'count'], 'column3': 'sum'})
df1.columns = list(map(''.join, df1.columns.values))

if not os.path.exists(path + "Data/"+folderName):
    os.makedirs(path + "Data/"+folderName)
pickledir = str(Path(os.getcwd()).parents[0]) + \
    '/data.cache/folder.name/{}/'.format(config.model_code)
Path(pickledir).mkdir(parents=True, exist_ok=True)

if not os.path.isfile("./df_all_data.pkl"):
    df_5_min.to_pickle("./df_all_data.pkl")
else:
    df_all = pd.read_pickle("./df_all_data.pkl")
    if df_5_min['test_date_time'].min() - df_all['test_date_time'].max() > datetime.timedelta(hours=1):
        df_5_min.to_pickle("./df_all_data.pkl")
    else:
        df_all = pd.concat([df_all, df_5_min]).drop_duplicates(
            subset=['slider_id'], keep='first').tail(1_000_000)
        df_all.to_pickle("./df_all_data.pkl")


def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            # we are interested in absolute coeff value
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr


corr_features = correlation(X_train, 0.8)
print('correlated features: ', len(set(corr_features)))


# Select k as per your business understanding
fs = SelectKBest(score_func=f_regression, k=30)
# Apply feature selection
fit = fs.fit(X, y)

# Wrapper method
sfs1 = SFS(  # knn(n_neighbors=3),
           # rfc(n_jobs=8),
           LGR(max_iter=1000),
           k_features='best',
           forward=True,
           floating=False,
           verbose=2,
           # scoring = 'neg_mean_squared_error',  # sklearn regressors
           scoring='accuracy',  # sklearn classifiers
           cv=0)

sfs1 = sfs1.fit(X, y, custom_feature_names=feature_names)

# hyperparameter using Optune


def objective(trial):

    rf_n_estimators = trial.suggest_int("rf_n_estimators", 100, 1000)
    rf_criterion = trial.suggest_categorical(
        "rf_criterion", ['gini', 'entropy'])
    rf_max_depth = trial.suggest_int("rf_max_depth", 1, 4)
    rf_min_samples_split = trial.suggest_float("rf_min_samples_split", 0.01, 1)

    model = RandomForestClassifier(
        n_estimators=rf_n_estimators,
        criterion=rf_criterion,
        max_depth=rf_max_depth,
        min_samples_split=rf_min_samples_split,
    )

    score = cross_val_score(model, X_train, y_train, cv=3)
    accuracy = score.mean()
    return accuracy


study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.RandomSampler(),
)


study.optimize(objective, n_trials=5)

# Feature engineering - Mutual Information

mi = mutual_info_classif(X_train, y_train)
mi = pd.Series(mi)
mi.index = X_train.columns
mi.sort_values(ascending=False).plot.bar(figsize=(20, 6))
plt.ylabel('Mutual Information')

df.interpolate()  # fill the missing value


def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_regression(
        X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


mi_scores = make_mi_scores(X, y, discrete_features)
mi_scores[::3]


def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")


plt.figure(dpi=100, figsize=(8, 5))
plot_mi_scores(mi_scores)

# Correlation
df_corr = df
cat_columns = list(df_corr.select_dtypes(include=['object']).columns)
df_corr[cat_columns] = df_corr[cat_columns].apply(
    lambda x: x.astype('category').cat.codes)
plt.figure(figsize=(15, 8))
corrMatrix = df_corr.corr()
sns.heatmap(corrMatrix, annot=True, fmt='.0%')

cor_matrix = df.corr().abs()
upper_tri = cor_matrix.where(
    np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper_tri.columns if any(
    upper_tri[column] > 0.95)]
df1 = df.drop(df.columns[to_drop], axis=1)

# correlation more than 0.8 will be removed

correlated_features = set()
correlation_matrix = df.corr()
for i in range(len(correlation_matrix .columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.8:
            colname = correlation_matrix.columns[i]
            correlated_features.add(colname)

y = df['column_name']
one_hot = MultiLabelBinarizer()
one_hot.fit_transform(y)
one_hot.classes_  # view classes

# Load the regression dataset
data = datasets.load_diabetes()
X, y = data['data'], data['target']

# Create a list of the feature names
features = np.array(data['feature_names'])

# Instantiate the visualizer
visualizer = FeatureCorrelation(labels=features)

visualizer.fit(X, y)        # Fit the data to the visualizer
visualizer.show()

# PCA - Principal Component Analysis https://www.kaggle.com/ryanholbrook/principal-component-analysis
# Create principal components
pca = PCA(n_components=2)  # the number of features
X_pca = pca.fit_transform(X_scaled)  # normalize the value (standardScaler)
component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
X_pca = pd.DataFrame(X_pca, columns=component_names)

print(pca.components_)  # variance value for each components
print(pca.explained_variance_)
# https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html

pca = PCA(n_components=2)
X_train = pca.fit_transform(X_train)  # standardize X_train and X_test
X_test = pca.transform(X_test)

kpca = KernelPCA(n_components=2, kernel='rbf')
X_train = kpca.fit_transform(X_train)
X_test = kpca.transform(X_test)

# Target Encoding

# Create the encoder instance. Choose m to control noise.
encoder = MEstimateEncoder(cols=["Zipcode"], m=5.0)

# Fit the encoder on the encoding split.
encoder.fit(X_encode, y_encode)

# Encode the Zipcode column to create the final training data
X_train = encoder.transform(X_pretrain)


# Preprocessing
mode_binary = Pipeline([
    ('encoder', SimpleImputer(strategy='most_frequent')),
    ('binary', BinaryEncoder())])

transformer = ColumnTransformer([
    ('one hot', OneHotEncoder(handle_unknown='ignore'), [
     'hotel', 'meal', 'market_segment', 'distribution_channel', 'reserved_room_type', 'deposit_type', 'customer_type']),
    ('mode binary', mode_binary, ['country']),
    ('impute mode', SimpleImputer(strategy='most_frequent'), ['children'])], remainder='passthrough')

# https://scikit-learn.org/stable/modules/compose.html#columntransformer-for-heterogeneous-data

X = df.drop('is_canceled', axis=1)
y = df['is_canceled']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=1515)

# Feature Scaling after split
# Fit and transfrom only for training data and transform for only test data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Or

sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

# Modelling
# KNN add MinMaxScaler
logreg = LogisticRegression()
tree = DecisionTreeClassifier(random_state=1515)
knn = KNeighborsClassifier()

logreg_pipe = Pipeline([('transformer', transformer), ('logreg', logreg)])
tree_pipe = Pipeline([('transformer', transformer), ('tree', tree)])
knn_pipe = Pipeline([('transformer', transformer),
                    ('scale', MinMaxScaler()), ('knn', knn)])


def model_evaluation(model, metric):
    model_cv = cross_val_score(
        model, X_train, y_train, cv=StratifiedKFold(n_splits=5), scoring=metric)
    return model_cv


logreg_pipe_cv = model_evaluation(logreg_pipe, 'precision')
tree_pipe_cv = model_evaluation(tree_pipe, 'precision')
knn_pipe_cv = model_evaluation(knn_pipe, 'precision')

for model in [logreg_pipe, tree_pipe, knn_pipe]:
    model.fit(X_train, y_train)

score_mean = [logreg_pipe_cv.mean(), tree_pipe_cv.mean(), knn_pipe_cv.mean()]
score_std = [logreg_pipe_cv.std(), tree_pipe_cv.std(), knn_pipe_cv.std()]
score_precision_score = [precision_score(y_test, logreg_pipe.predict(X_test)), precision_score(
    y_test, tree_pipe.predict(X_test)), precision_score(y_test, knn_pipe.predict(X_test))]
method_name = ['Logistic Regression',
    'Decision Tree Classifier', 'KNN Classifier']
cv_summary = pd.DataFrame({
    'method': method_name,
    'mean score': score_mean,
    'std score': score_std,
    'precision score': score_precision_score})

cv_summary

# Hyperparamer tuning
estimator = Pipeline([
    ('transformer', transformer),
    ('model', logreg)])

hyperparam_space = {
    'model__C': [1, 5, 10, 20, 30, 50],
    'model__class_weight': ['dict', 'balanced'],
    'model__solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
    'model__max_iter': [50, 100, 150, 200, 300],
    'model__random_state': [1515],
    'model__n_jobs': [-1]
}

random = RandomizedSearchCV(
                estimator,
                param_distributions=hyperparam_space,
                cv=StratifiedKFold(n_splits=5),
                scoring='precision',
                n_iter=10,
                n_jobs=-1)

random.fit(X_train, y_train)

print('best score', random.best_score_)
print('best param', random.best_params_)

# Rerun after Hyperparamer tuning
estimator.fit(X_train, y_train)
y_pred_estimator = estimator.predict(X_test)
before = precision_score(y_test, y_pred_estimator)

random.best_estimator_.fit(X_train, y_train)
y_predict = random.best_estimator_.predict(X_test)
after = precision_score(y_test, y_predict)

score_list = [before, after]
method_name = ['Logistic Regression Before Tuning',
    'Logistic Regression After Tuning']
best_summary = pd.DataFrame({
    'method': method_name,
    'score': score_list
})
print(best_summary)

# Feature importance


X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=3,
    n_redundant=0,
    n_repeated=0,
    n_classes=2,
    random_state=0,
    shuffle=False,
)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=42)
feature_names = [f"feature {i}" for i in range(X.shape[1])]
forest = RandomForestClassifier(random_state=0)
forest.fit(X_train, y_train)

importances = forest.feature_importances_
std = np.std(
    [tree.feature_importances_ for tree in forest.estimators_], axis=0)
forest_importances = pd.Series(importances, index=feature_names)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()

feature_names

forest_importances.sort_values(ascending=False)

kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Random forest
# https://chrisalbon.com/machine_learning/trees_and_forests/random_forest_classifier_example/
# Load the library with the iris dataset

# Set random seed
np.random.seed(0)

# Create an object called iris with the iris data
iris = load_iris()

# Create a dataframe with the four feature variables
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add a new column with the species names,
#  this is what we are going to try to predict
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Create a new column that for each row,
#  generates a random number between 0 and 1, and
# if that value is less than or equal to .75,
#  then sets the value of that cell as True
# and false otherwise.
#  This is a quick and dirty way of randomly assigning some rows to
# be used as the training data and some as the test data.
df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75

# Create two new dataframes, one with the training rows, one with the test rows
# train, test = df[df['is_train'] == True]
# df[df['is_train'] == False]

if df['is_train'] is True:
    train = df[df['is_train']]

elif df['is_train'] is False:
    test = df[df['is_train']]

# Show the number of observations for the test and training dataframes
print('Number of observations in the training data:', len(train))
print('Number of observations in the test data:', len(test))

# Create a list of the feature column's names
features = df.columns[:4]

# train['species'] contains the actual species names. Before we can use it,
# we need to convert each species name into a digit. So, in this case there
# are three species, which have been coded as 0, 1, or 2.
y = pd.factorize(train['species'])[0]

# Create a random forest Classifier. By convention, clf means 'Classifier'
clf = RandomForestClassifier(n_jobs=2, random_state=0)

# Train the Classifier to take the training features and learn how they relate
# to the training y (the species)
clf.fit(train[features], y)

# Apply the Classifier we trained to the test data
#  (which, remember, it has never seen before)
clf.predict(test[features])

# View the predicted probabilities of the first 10 observations
clf.predict_proba(test[features])[0:10]

# Create actual english names for the plants for each predicted plant class
preds = iris.target_names[clf.predict(test[features])]

# View the PREDICTED species for the first five observations
preds[0:5]

# View the ACTUAL species for the first five observations
test['species'].head()

# Create confusion matrix
pd.crosstab(test['species'], preds,
            rownames=['Actual Species'], colnames=['Predicted Species'])

# View a list of the features and their importance scores
list(zip(train[features], clf.feature_importances_))


fuzz.ratio("this is a coding", "this is code")
fuzz.partial_ratio("this is a coding", "this is code")
fuzz.token_sort_ratio("this is a coding", "this is code")
fuzz.token_set_ratio("this is a coding", "this is code")

fuzzywuzzy.process.extract("south korea", countries,
                           limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)


def replace_matches_in_column(df, column, string_to_match, min_ratio=47):
    # get a list of unique strings
    strings = df[column].unique()

    # get the top 10 closest matches to our input string
    matches = fuzzywuzzy.process.extract(string_to_match, strings,
                                         limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)


# Connection postgresql, encoding password
urllib.parse.quote_plus("password")
https: // docs.sqlalchemy.org/en/14/core/engines.html

USERNAME = 'email@email.com'
PASSWORD = 'password'
engine = create_engine('postgresql://' + USERNAME + ':' +
                       PASSWORD + 'locationof_the_database')
column_1_value = 'example_%%'  # double percentage is necassary due to python syntax
Columns_2_value = 'example'
sql = f"select * FROM d.database WHERE column_1 like '{column_1_value}' and column_2 = '{Columns_2_value}' limit 10"
pd.read_sql(sql, engine)
engine.dispose()

try:
    function()
except Exception:
    function()
    pass

a = []
for i in range(5):
    a.append(i)
print(a)

# Scatter plot
colors = np.where(((df["column1"] <= -0.1) | (df["column2"]
                  >= 0.1)) & (df['column3'] >= 20), 'r', 'k')
df.reset_index().plot.scatter(x='column_x|datetime', y='column_y', s=100, c=colors)
plt.gcf().set_size_inches((20, 18))
[plt.axhline(y=i, color='y', linestyle='-')
             for i in [0.1, -0.1]]  # draw a line at y axis
# plt.show()

mpl.rc('figure', max_open_warning=0)  # ignore warning

# Line Plot
df.plot.line(x='column1', y='column2')
df.query('column1 == "someString" & column2 == "someString"'
                    ).plot(kind='line', x=['column3', 'column4', 'column5'], y='column4', figsize=(18, 10), marker='o', markerfacecolor='red',
                     label=['column3', 'column4', 'column5']  # or label=value)
[plt.pyplot.axhline(y=i, color='y', linestyle='-')
                    for i in [0.1, -0.1]]  # a stright line at y axis
[plt.pyplot.axhline(y=i, color='y', linestyle='-')
                    for i in [0]]  # second stright line at y axis in the same fig
plt.legend()
# bar plot

df.plot(kind='bar')

df[(df['column1'] == 'some_variable')].plot(kind='scatter',
    x='column_date_time', y='column2', s=10, c='r', figsize=(12, 6))
plt.axvline(pd.Timestamp('2022-01-10 09:40:28'), color='b')

fig, ax=plt.subplots()
for name, group in df.groupby('columntobegrouped'):
    group.plot('datetime_column', y='column_relatedto_columntobegrouped',
               label=name,  ax=ax, figsize=(18, 10))

import seaborn as sns
# https://seaborn.pydata.org/generated/seaborn.scatterplot.html
sns.set(rc={"figure.figsize": (18, 8)})
sns.set_theme(style="whitegrid")  # or
sns.set_style("dark")  # {darkgrid, whitegrid, dark, white, ticks} options
ax=sns.lineplot(data=df, x='datetime_column', y='value_column', hue='multiple_value_column',
             style="multiple_value_column", markers=True, dashes=False, err_style="bars", ci=68, legend="full")

ax=sns.scatterplot(data=df, x="timecolumn", y="column2",
                   hue="column3 -must categorical")
ax=sns.barplot(x="column1", y="column2", hue="column3", data=df)


# return 2 values from a function, generate documentation by using vs code extension "mintlify doc writer"
def afunction(a, b):
    """
    It takes two numbers, adds them together, and returns the sum and the second number

    :param a: a number
    :param b: a list of numbers
    :return: c, b
    """
    c=a + b
    return c, b
a, b=afunction()

if i > 3 and i == 4:
    print('AND')

if i > 3 & i == 4:
    print('OR')

# pandas sample for testing
df=pd.DataFrame({'num_legs': [2, 4, 8, 0],
                   'num_wings': [2, 0, 0, 0],
                   'num_specimen_seen': [10, 2, 1, 8]},
                  index=['falcon', 'dog', 'spider', 'fish'])

data=[['tom', 10], ['nick', 15], ['juli', 14]]
df=pd.DataFrame(data, columns=['Name', 'Age'])

# df to test the time for execution time
df=pd.DataFrame(np.random.randn(40000, 3), columns=['col1', 'col2', 'col3'])

# check list for more than 1
mylist=['A', 'A', 'B', 'A', 'D', 'E', 'D', 'B', 'B']

if 'B' in set([i for i in mylist if mylist.count(i) >= 3]):
    print('yes')

from apyori import apriori
rules=apriori(transactions=df, min_support=0.003,
              min_confidence=0.2, min_lift=3, min_length=2, max_length=2)

# creating emtpy dataframe with one column and 2 sample rows
pd.DataFrame({'unique_value': [1, 2]})

# config.py #file stored in the same folder of below code
RTPMserver="00.00.00.00"
RTPMAccount="username"
RTPMPass="password"
RTPMschema='schema_name'

import config
import pymysql
pymysql.install_as_MySQLdb()
import MySQLdb
databasecon=MySQLdb.connect(host=config.RTPMserver,
                          port=3309,
                          user=config.RTPMAccount,
                          passwd=config.RTPMPass,
                          db=config.RTPMschema,
                          charset="utf8")
sql="select *\
        from database limit 10"
df=pd.read_sql(sql, databasecon)
con.close()
https: // dev.mysql.com/doc/connector-python/en/connector-python-example-cursor-transaction.html

with databasecon.cursor() as cursor:
    sql="INSERT INTO data_table (column1, column2, column3) \
    VALUES ('LEC 4178', NOW()--datetime, 'DAO-YP2')"
    cursor.execute(sql)
rttccon.commit()
rttccon.close()

df.to_pickle("./df_name.pkl")  # to store the dataframe
pd.read_pickle("./df_name.pkl")  # read the pickle back

# basic pickle
mylist=[1, 2, 3, '4']
with open('sample.pkl', 'wb') as f:
    pickle.dump(mylist, f)
with open('sample.pkl', 'rb') as f:
    mynewlist=pickle.load(f)
mynewlist

# pickle one liner
mylist=[1, 2, 3, 4, '5']
pickle.dump(mylist, open('sample_file.pkl', 'wb'))
mynewlist=pickle.load(open('sample_file.pkl', 'rb'))
mynewlist

from collections import deque
lst=deque([], maxlen=4)  # empty deque
lst

import timeit
import sys

def for_loop(n=1_000):
    s=0
    for i in range(n):
        s += i
    return s

def main():
    print(" ", timeit.timeit(for_loop, number=1))

if __name__ == "__main__":
    main()
    # sys.exit(main()) https://docs.python.org/3/library/__main__.html

from time import sleep

while True:
  localtime=time.localtime()
  result=time.strftime("%I:%M:%S %p", localtime)
  print(result, end="", flush=True)
  print("\r", end="", flush=True)
  sleep(5)  # s

# Handling exception
from contextlib import suppress

with suppress(IDontLikeYouException, YouAreBeingMeanException):
     do_something()

try:
    1/0
except (NameError, ZeroDivisionError):
    pass
except (IDontLikeYouException, YouAreBeingMeanException) as e:
    pass

# to check if dataframe is empty
if df.empty:
    print('DataFrame is empty!')

len(df.index) == 0

s.mask(s > 0)

# Rounding the float to 2 decimal points
df['column1'].round(2)
for row in df.itertuples():
    print(round(row.column1, 2))

# Example Pandas dataframe

pd.DataFrame({'col one': [100, 200], 'col two': [300, 400]})
pd.DataFrame(np.random.rand(4, 8), columns=list('abcdefgh'))
pd.util.testing.makeDataFrame().head()
pd.util.testing.makeMissingDataframe().head()
pd.util.testing.makeTimeDataFrame().head()
pd.util.testing.makeMixedDataFrame()
[x for x in dir(pd.util.testing) if x.startswith('make')]

from sklearn.feature_selection import VarianceThreshold
X=[[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]]
sel=VarianceThreshold(threshold=(.8 * (1 - .8)))
sel.fit_transform(X)
sum(sel.get_support())  # sum of features that not quasi-constant


# full training without split
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, roc_auc_score, precision_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
iris=load_iris()
data=iris.data
target=iris.target
names=iris.target_names
df=pd.DataFrame(data, columns=iris.feature_names)
df['species']=iris.target
# df['species']=df['species'].replace(to_replace=[0, 1, 2], value=['setosa', 'versicolor', 'virginica']) # Replace with name
X=df.drop(['species'], axis=1)
y=df['species']
scaler=StandardScaler()
X_std=scaler.fit_transform(X)
clf=LogisticRegression(random_state=0, class_weight="balanced")
model=clf.fit(X_std, y)
new_observation=[[.5, .5, .5, .5]]
model.predict(new_observation)
model.predict_proba(new_observation)
print(model.score(X_std, y))
pd. concat([pd.DataFrame(model.predict_proba(X_std).max(1), columns=['proba_1']),
            pd.DataFrame(model.predict(X_std), columns=['prediction']),
           y.rename("passfail")], axis=1)

# Sample of ROC
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X, y=make_classification(n_samples=10000,
                           n_features=10,
                           n_classes=2,
                           n_informative=3,
                           random_state=3)

X_train, X_test, y_train, y_test=train_test_split(
    X, y, test_size=0.1, random_state=1)
clf=LogisticRegression()
clf.fit(X_train, y_train)
y_score=clf.predict_proba(X_test)[:, 1]
false_positive_rate, true_positive_rate, threshold=roc_curve(y_test, y_score)
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0], c=".7"), plt.plot([1, 1], c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda=LinearDiscriminantAnalysis(n_components=1)
X_lda=lda.fit(X, y).transform(X)
lda.explained_variance_ratio_

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda=LDA(n_components=2)
X_train=lda.fit_transform(X_train, y_train)
X_test=lda.transform(X_test)

# create a new path
from pathlib import Path
cwd=Path.cwd()
joined_path=cwd / 'Output'
joined_path.mkdir(exist_ok=True)

# Create a perceptron object with the parameters: 40 iterations (epochs) over the data, and a learning rate of 0.1
ppn=Perceptron(n_iter=40, eta0=0.1, random_state=0)

# Train the perceptron
ppn.fit(X_train_std, y_train)

# detecting outlier
from sklearn.covariance import EllipticEnvelope
# Create detector
outlier_detector=EllipticEnvelope(contamination=.1)

# Fit detector
outlier_detector.fit(X)

# Predict outliers
outlier_detector.predict(X)

# corr
corrmat=X_train.corr()
corrmat=corrmat.abs().unstack()  # absolute value of corr coef
corrmat=corrmat.sort_values(ascending=False)
corrmat=corrmat[corrmat >= 0.8]
corrmat=corrmat[corrmat < 1]
corrmat=pd.DataFrame(corrmat).reset_index()

# Univariate feature selection
from sklearn.feature_selection import f_classif, f_regression
univariate=f_classif(X_train, y_train)

# helper.py
# -------------------------------------
import logging
logger=logging.getLogger(__name__)
logger.info('HELLO')

# main.py
# -------------------------------------
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(name)s - %(levelname)s - %(message)s')
import helper

# logging.conf or logging.ini
[loggers]
keys=root, simpleExample

[handlers]
keys=consoleHandler

[formatters]
keys=simpleFormatter

[logger_root]
level=DEBUG
handlers=consoleHandler

[logger_simpleExample]
level=DEBUG
handlers=consoleHandler
qualname=simpleExample
propagate=0

[handler_consoleHandler]
class=StreamHandler
level=DEBUG
formatter=simpleFormatter
args=(sys.stdout,)

[formatter_simpleFormatter]
format=% (asctime)s - %(name)s - %(levelname)s - %(message)s

# Lambda
def myfunc(n):
    return lambda x: x * n

doubler=myfunc(2)
print(doubler(6))  # 12

tripler=myfunc(3)
print(tripler(6))  # 18

# minimal example for own exception class
class ValueTooHighError(Exception):
    pass

# or add some more information for handlers
class ValueTooLowError(Exception):
    def __init__(self, message, value):
        self.message=message
        self.value=value

def test_value(a):
    if a > 1000:
        raise ValueTooHighError('Value is too high.')
    if a < 5:
        # Note that the constructor takes 2 arguments here
        raise ValueTooLowError('Value is too low.', a)
    return a

try:
    test_value(1)
except ValueTooHighError as e:
    print(e)
except ValueTooLowError as e:
    print(e.message, 'The value is:', e.value)

import json
person={"name": "John", "age": 30, "city": "New York",
    "hasChildren": False, "titles": ["engineer", "programmer"]}
person_json=json.dumps(person)


import heapq
test_list=[1, 2, 3, 50, 60, 70]
print(heapq.nlargest(3, test_list))
print(heapq.nsmallest(3, test_list))


df=pd.read_csv('spaceship_imbalanced.csv')
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
resamp=RandomUnderSampler()
balX, baly=resamp.fit_resample(X, y)
sns.countplot(x=baly)
plt.show()

data.text=b'Code is like humor. When you have to explain it, its bad.'
f=open("data.txt", "rb")
# sets Reference point to tenth
# position to the left from end
f.seek(-10, 2)
# prints current position
print(f.tell())
# Converting binary to string and
# printing
print(f.readline().decode('utf-8'))
f.close()

string='This is initial string.'
file=StringIO(string)
print("Is the file stream interactive?", file.isatty())
print("Is the file stream readable?", file.readable())
print("Is the file stream writable?", file.writable())
print("Is the file stream seekable?", file.seekable())
print("Is the file closed?", file.closed)
print(file.read())
file.write(" Welcome to geeksforgeeks.")
print(file.tell())
print(file.getvalue())
file.seek(0)

print(file.tell())
print('The string after writing is:', file.read())
file.seek(0)
file.truncate(22)
print(file.read())

file.close()
print("Is the file closed?", file.closed)

init_string=b'hello'
in_memory=io.BytesIO(init_string)
# in_memory.seek(0, 2)
in_memory.write(b' world')
in_memory.seek(0)
print(in_memory.read())
string='This is initial string.'
file=StringIO(string)
# print(file.read())
print(file.getvalue())
file.write('world')
print(file.read())
print(file.getvalue())

import boto3

s3_resource=boto3.resource('s3',
                             endpoint_url=config.endpoint_url,
                             aws_access_key_id=config.aws_access_key_id,
                             aws_secret_access_key=config.aws_secret_access_key)

my_bucket=s3_resource.Bucket(config.bucket)
my_bucket.Object(img)

def s3_con(aaki, asak, endpoint_url):
    session=boto3.session.Session()
    s3_client=session.client(
    service_name='s3',
    aws_access_key_id=aaki,
    aws_secret_access_key=asak,
    endpoint_url=endpoint_url
    )
    return s3_client
aws_access_key_id='keyid'
aws_secret_access_key='secretid'
endpoint_url='http://s3.xx.xx.com'
bucket='bucket_name'

s3_client=s3_con(aws_access_key_id, aws_secret_access_key, endpoint_url)
for obj in s3_client.list_objects(Bucket=bucket, Prefix='mho_path/images/some_path')['Contents']:
#     print(obj)
    try:
        print(obj['Key'])
#         s3_client.download_file(bucket,obj['Key'],file_path+obj['Key'].rsplit('/', 1)[1])
    except:
        continue

for obj in list(my_bucket.objects.filter(Prefix=prefix)):
    print(obj.key)
    image=my_bucket.Object(obj.key)
    img_data=image.get()['Body'].read()
    img=Image.open(io.BytesIO(img_data))
    plt.imshow(img)
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import time

pause_time=0.2  # seconds between frames
a=np.random.rand(3, 3)

for t in range(0, 10):
    plt.imshow(a)
    plt.title(t)
    display.display(plt.gcf())
    display.clear_output(wait=True)
    time.sleep(pause_time)

    a=np.random.rand(3, 3)

s3_resource=boto3.resource('s3',
                             endpoint_url='http://s3.xx.com',
                             aws_access_key_id='xxx',
                             aws_secret_access_key='xxx')

my_bucket=s3_resource.Bucket('bucket_name')
prefix='mho_path/images/ph_name/path_name'
files=list(my_bucket.objects.filter(Prefix=prefix))
files  # list all files

for obj in list(my_bucket.objects.filter(Prefix=prefix)):  # iterate and show all images
    print(obj.key)
    image=my_bucket.Object(obj.key)
    img_data=image.get()['Body'].read()
    img=Image.open(io.BytesIO(img_data))
    plt.imshow(img)
    plt.show()

def write_image_to_s3(img_array, bucket, key, region_name='ap-southeast-1'):
    """Write an image array into S3 bucket

    Parameters
    ----------
    bucket: string
        Bucket name
    key : string
        Path in s3

    Returns
    -------
    None
    """
    s3=boto3.resource('s3', region_name)
    bucket=s3.Bucket(bucket)
    object=bucket.Object(key)
    file_stream=BytesIO()
    im=Image.fromarray(img_array)
    im.save(file_stream, format='jpeg')
    object.put(Body=file_stream.getvalue())


import cv2
import numpy as np
from PIL import Image

pil_image=Image.open("demo2.jpg")  # open image using PIL

# use numpy to convert the pil_image into a numpy array
numpy_image=numpy.array(pil_img)

# convert to a openCV2 image, notice the COLOR_RGB2BGR which means that
# the color is converted from RGB to BGR format
opencv_image=cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

import cv2
import numpy as np
from PIL import Image

opencv_image=cv2.imread("demo2.jpg")  # open image using openCV2

# convert from openCV2 to PIL. Notice the COLOR_BGR2RGB which means that
# the color is converted from BGR to RGB
color_coverted=cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
pil_image=Image.fromarray(color_coverted)

from sklearn.metrics import classification_report
import pandas as pd
y_true=[0, 1, 2, 2, 2]
y_pred=[0, 0, 2, 2, 1]
target_names=['class 0', 'class 1', 'class 2']
dft=classification_report(
    y_true, y_pred, target_names=target_names, output_dict=True)
pd.DataFrame(dft).transpose().round(2)


from smb.SMBConnection import SMBConnection

conn=SMBConnection(username="svcv@test.com",
                     password="$password",
                     my_name="my_name",
                     remote_name="remote_name")

conn.connect('ip or remote name')
for img_path in favi_path_df['new_path'][:1]:
    print(img_path)

    buffer=io.BytesIO()
    conn.retrieveFile(service_name="J$",
                      path=img_path, file_obj=buffer)
    buffer.seek(0)
    img=buffer.read()
    decoded=cv2.imdecode(np.frombuffer(img, np.uint8), -1)
    print('OpenCV:\n', decoded)
    plt.imshow(decoded)
    plt.show()
    # image = np.array(Image.open(io.BytesIO(img))) #or this one
    # print('PIL:\n', image)
    # plt.imshow(image)
    # plt.show()
    buffer.close()


# API
import base64, requests, json

# Step 1: open the images on your local position (If you read the image from memory, it is another topic)
with open( 'image_sample.jpg' , "rb") as f:
    img_bytes=f.read()

# Step 2: Encode the bytes data to serializable data, then transfer the python type to string format
img_str=base64.b64encode(img_bytes).decode("utf-8")

# Step 3: Compile the data into python dict, and transfer to json format(**Notice, cannot use str, please make sure you use json.dumps)
data_str=json.dumps({
    'img_name': "image_sample",
    "img_str": img_str
})

# Step 4: Call the dataIku API service and get response
res=requests.post(
    url= 'http://internal-',
    data=data_str
)

# Step 5: Parse the response to the result you can use
result=res.json()
res.close()

#connecting to sql server
server = "10.00.000.000"
database = "K2_db_name"
username = "user"
password = "password"
conn_str = f"DRIVER={{SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}"
conn = pyodbc.connect(conn_str)
cursor = conn.cursor()

cursor.execute("SELECT TOP 10 Image2D FROM TB_Image2D")
rows = cursor.fetchall()

for row in rows:
    print(row)
    image_stream = io.BytesIO(row[0])
    display(Image(data=image_stream.getvalue()))
cursor.close()
conn.close()


import pymssql
server = "10.00.000.000"
port = 1433
password = "password"
user = "user"
query = "SELECT TOP 10 Image2D FROM K2_db_name.dbo.table_name"

conn = pymssql.connect(server, user, password, port=port)
cursor = conn.cursor(as_dict=True)
cursor.execute(query)
rows = cursor.fetchall()
for row in rows:
    image_stream = io.BytesIO(row)
    display(Image(data=image_stream.getvalue()))
cursor.close()
conn.close()


#list all the file 
import os
path_input = 'report'
for file in os.scandir(path_input):
    if file.is_file() and file.name.endswith(".xlsx"):
        print(file.name)

# request template
# https://github.com/public-apis/public-apis
import requests
import json
url="https://cat-fact.herokuapp.com/facts"
# url = "https://api.quran.com/api/v4/chapters/2?language=en"
r=requests.get(url)
# data=r.json()
data=json.loads(r.text)
print(data)

import sys
import time
import logging
from watchdog.observers import Observer
from watchdog.events import LoggingEventHandler

if __name__ == "__main__":
    # Set the format for logging info
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    # Set format for displaying path
    path=sys.argv[1] if len(sys.argv) > 1 else '.'

    # Initialize logging event handler
    event_handler=LoggingEventHandler()

    # Initialize Observer
    observer=Observer()
    observer.schedule(event_handler, path, recursive=True)

    # Start the observer
    observer.start()
    try:
        while True:
            # Set the thread sleep time
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

# from zipfile import ZipFile
with ZipFile("PDF for Global\P001-003300-240G.zip", "r") as zip_ref:
    zip_ref.extractall("PDF for Global/target_dir")

# compare two csv files
with open('test1.csv', 'r') as t1, open('test2.csv', 'r') as t2:
    fileone=t1.readlines()
    filetwo=t2.readlines()

with open('update.csv', 'w') as outFile:
    for line in filetwo:
        if line not in fileone:
            outFile.write(line)
