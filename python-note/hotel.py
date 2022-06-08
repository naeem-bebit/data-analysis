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

# %matplotlib inline
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)  # Set columns mac without truncated
pd.set_option('display.max_rows', None)  # Set display max

with pd.option_context('display.max_colwidth', None):  # showing max column
  display(df)

# PreProcessing

# Splitting Data

# Modeling

df[~df['Hotel'].str.contains('CZK|Hyatt.com|Trip.com|Hotels.com|Sponsored|Booking.com|Agoda.com|Expedia.com|FindHotel').fillna(
    False)].reset_index(drop=True)

df = pd.read_csv('https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv',
                 usecols=['hotel', 'is_canceled', 'adults', 'children', 'babies', 'meal', 'country', 'market_segment', 'distribution_channel', 'reserved_room_type', 'booking_changes', 'deposit_type', 'days_in_waiting_list', 'customer_type', 'required_car_parking_spaces', 'total_of_special_requests'])
df.head()

# set for large data or assign data type to the particular columns
df.pd.read_csv('file_name.csv', low_memory=False)

df.shape  # Size of the dataframe

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

df.dtypes  # Get the type of the column data
# Get the list of object category
list(df.select_dtypes(include=['object']).columns)

df['Target'] = np.where(df['column_name'].isin(
    df1['column_name']), 'Fail', 'Pass')  # create failed pass column

df.drop(['columns2', 'column1'], axis=1)

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

# showing Column_name3 and groupby by column_name1 & column_name2
df.groupby(['Column_name1', 'Column_name2'])['Column_name3'].sum()

df1 = df.groupby(['column_name1', 'column_name2'], as_index=False).agg(
    {'column_datetime': ['min', 'count'], 'column3': 'sum'})
df1.columns = list(map(''.join, df1.columns.values))

# Feature engineering - Mutual Information

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

# df to test the time for execution time
df=pd.DataFrame(np.random.randn(40000, 3), columns=['col1', 'col2', 'col3'])

# check list for more than 1
mylist=['A', 'A', 'B', 'A', 'D', 'E', 'D', 'B', 'B']

if 'B' in set([i for i in mylist if mylist.count(i) >= 3]):
    print('yes')

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
