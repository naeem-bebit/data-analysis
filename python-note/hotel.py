import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None) # Set columns mac without truncated
pd.set_option('display.max_rows', None) # Set display max

# PreProcessing
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from category_encoders import BinaryEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Splitting Data
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV

# Modeling
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score

df = pd.read_csv('https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv',
                 usecols = ['hotel', 'is_canceled', 'adults', 'children', 'babies', 'meal', 'country', 'market_segment', 'distribution_channel', 'reserved_room_type', 'booking_changes', 'deposit_type', 'days_in_waiting_list', 'customer_type', 'required_car_parking_spaces', 'total_of_special_requests'])
df.head()

df.shape # Size of the dataframe

df.info() # Basic info of the table such as type

df.describe() # Distribution data

df.isnull().sum() #Row with NaN value

df.isnull().sum().sum() # Total NaN value

df[df.isnull().T.any()] #|| df[df.isnull().any(axis=1)] #Rows with NaN value

df['hotel'].value_counts().plot.bar() #count the main column value and plot

pd.crosstab(df['hotel'], df['is_canceled'], margins=True, margins_name = 'Total')
pd.crosstab(df['adults'], df['children'], margins=True, margins_name = 'Total')

df.isna().sum()/len(df)*100 # Percentage of NaN value

df.dtypes #Get the type of the column data
list(df.select_dtypes(include=['object']).columns) # Get the list of object category

# Run the processing using the categorical preprocessing
cat_columns = list(df.select_dtypes(include=['object']).columns)
df[cat_columns] = df[cat_columns].apply(lambda x: x.astype('category').cat.codes)

# Correlation
df_corr = df
cat_columns = list(df_corr.select_dtypes(include=['object']).columns)
df_corr[cat_columns] = df_corr[cat_columns].apply(lambda x: x.astype('category').cat.codes)
plt.figure(figsize=(15,8))
corrMatrix = df_corr.corr()
sns.heatmap(corrMatrix, annot= True, fmt='.0%')

# Preprocessing
mode_binary = Pipeline([
    ('encoder', SimpleImputer(strategy = 'most_frequent')),
    ('binary', BinaryEncoder())])

transformer = ColumnTransformer([
    ('one hot', OneHotEncoder(handle_unknown = 'ignore'), [ 'hotel', 'meal', 'market_segment', 'distribution_channel', 'reserved_room_type', 'deposit_type', 'customer_type']),
    ('mode binary', mode_binary, ['country']),
    ('impute mode', SimpleImputer(strategy = 'most_frequent'), ['children'])], remainder = 'passthrough')

#https://scikit-learn.org/stable/modules/compose.html#columntransformer-for-heterogeneous-data

X = df.drop('is_canceled', axis = 1)
y = df['is_canceled']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size = 0.2, random_state = 1515)

# Modelling
## KNN add MinMaxScaler
logreg = LogisticRegression()
tree = DecisionTreeClassifier(random_state = 1515)
knn = KNeighborsClassifier()

logreg_pipe = Pipeline([('transformer', transformer), ('logreg', logreg)])
tree_pipe = Pipeline([('transformer', transformer), ('tree', tree)])
knn_pipe = Pipeline([('transformer', transformer), ('scale', MinMaxScaler()), ('knn', knn)])

def model_evaluation(model, metric):
    model_cv = cross_val_score(model, X_train, y_train, cv = StratifiedKFold(n_splits = 5), scoring = metric)
    return model_cv

logreg_pipe_cv = model_evaluation(logreg_pipe, 'precision')
tree_pipe_cv = model_evaluation(tree_pipe, 'precision')
knn_pipe_cv = model_evaluation(knn_pipe, 'precision')

for model in [logreg_pipe, tree_pipe, knn_pipe]:
    model.fit(X_train, y_train)
    
score_mean = [logreg_pipe_cv.mean(), tree_pipe_cv.mean(), knn_pipe_cv.mean()]
score_std = [logreg_pipe_cv.std(), tree_pipe_cv.std(), knn_pipe_cv.std()]
score_precision_score = [precision_score(y_test, logreg_pipe.predict(X_test)), precision_score(y_test, tree_pipe.predict(X_test)), precision_score(y_test, knn_pipe.predict(X_test))]
method_name = ['Logistic Regression', 'Decision Tree Classifier', 'KNN Classifier']
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
                param_distributions = hyperparam_space,
                cv = StratifiedKFold(n_splits = 5),
                scoring = 'precision',
                n_iter = 10,
                n_jobs = -1)

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
method_name = ['Logistic Regression Before Tuning', 'Logistic Regression After Tuning']
best_summary = pd.DataFrame({
    'method': method_name,
    'score': score_list
})
print(best_summary)
