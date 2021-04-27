import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None) # Set columns mac without truncated
pd.set_option('display.max_rows', None) # Set display max

df = pd.read_csv('https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv')
df.head()

df.shape # Size of the dataframe

df.info() # Basic info of the table such as type

df.describe() # Distribution data

df.isnull().sum() #Row with NaN value

df.isnull().sum().sum() # Total NaN value

df[df.isnull().T.any()] || df[df.isnull().any(axis=1)] #Rows with NaN value

df['hotel'].value_counts().plot.bar() #count the main column value and plot



