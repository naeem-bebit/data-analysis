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

df[~df['Hotel'].str.contains('CZK|Hyatt.com|Trip.com|Hotels.com|Sponsored|Booking.com|Agoda.com|Expedia.com|FindHotel').fillna(False)].reset_index(drop=True)

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

df['column_1'].str.split('-',expand=True)[1].reset_index(drop=True) #spread the string column by '-' and select the second column
df['Time Zone'].str.split('/',expand=True)[[0,1]].reset_index(drop=True).rename(columns={0: 'continent', 1:'city'}) #spread and rename two columns

df.dtypes #Get the type of the column data
list(df.select_dtypes(include=['object']).columns) # Get the list of object category

df.query('column_name == "string_value"') #query

pd.merge(df1,df2, on="columns")

plt.figure(figsize=(15,8)) # Plot the data with threshold
df_count_country = df[3].value_counts()
df_count_country['other']=df_count_country[df_count_country < 10].sum()
df_count_country = df_count_country[df_count_country > 10]
df_count_country.plot(kind='bar')

df['column'].str.replace('UTC ', '') #remove UTC from the string in pandas column

df.groupby(['column 1', 'column 2']).size().reset_index(name='counts') # Groupby

df[df.duplicated()] #check for duplicate in any column
pd.to_datetime(df['Column']) #change column to datetime

df['Column Name'].value_counts().reset_index().rename(columns={'index': 'New_name'})

df[df['Column name'].notnull()] #find row with nan value
df[df['Column name'].isnull()] #find row with not nan value

df.columns = ['Column 1', 'Column 2','Column 3'] #rename all columns

[col[0] for col in df.columns] #multiindex column

df.to_csv('sme2.csv', encoding='utf-8') #save to csv



# Run the processing using the categorical preprocessing
cat_columns = list(df.select_dtypes(include=['object']).columns)
df[cat_columns] = df[cat_columns].apply(lambda x: x.astype('category').cat.codes)

df[~df['Pax Email'].isin(ex_email+scam_email)] #isin

df.rename(columns={"Destination": "iata_code"}) #rename the column

df.groupby(['Fruit','Name'])['Number'].sum()

#Feature engineering - Mutual Information
from sklearn.feature_selection import mutual_info_regression

def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
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
df_corr[cat_columns] = df_corr[cat_columns].apply(lambda x: x.astype('category').cat.codes)
plt.figure(figsize=(15,8))
corrMatrix = df_corr.corr()
sns.heatmap(corrMatrix, annot= True, fmt='.0%')

cor_matrix = df.corr().abs()
upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
df1 = df.drop(df.columns[to_drop], axis=1)

#correlation more than 0.8 will be removed

correlated_features = set()
correlation_matrix = paribas_data.corr()
for i in range(len(correlation_matrix .columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.8:
            colname = correlation_matrix.columns[i]
            correlated_features.add(colname)

## PCA - Principal Component Analysis https://www.kaggle.com/ryanholbrook/principal-component-analysis
from sklearn.decomposition import PCA

# Create principal components
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Convert to dataframe
component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
X_pca = pd.DataFrame(X_pca, columns=component_names)


## Target Encoding
from category_encoders import MEstimateEncoder

# Create the encoder instance. Choose m to control noise.
encoder = MEstimateEncoder(cols=["Zipcode"], m=5.0)

# Fit the encoder on the encoding split.
encoder.fit(X_encode, y_encode)

# Encode the Zipcode column to create the final training data
X_train = encoder.transform(X_pretrain)


## Preprocessing
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
