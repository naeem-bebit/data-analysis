"""K-Nearest Neighbors Classification."""
import pandas as pd
from sklearn import neighbors
import numpy as np
# %matplotlib inline
import seaborn

training_data = pd.DataFrame()
training_data['test_1'] = [0.3051, 0.4949, 0.6974,
                           0.3769, 0.2231, 0.341, 0.4436, 0.5897, 0.6308, 0.5]
training_data['test_2'] = [0.5846, 0.2654, 0.2615, 0.4538,
                           0.4615, 0.8308, 0.4962, 0.3269, 0.5346, 0.6731]
training_data['outcome'] = ['win', 'win', 'win', 'win',
                            'win', 'loss', 'loss', 'loss', 'loss', 'loss']

seaborn.lmplot('test_1', 'test_2', data=training_data,
               fit_reg=False, hue="outcome",
               scatter_kws={"marker": "D", "s": 100})

X = training_data.as_matrix(columns=['test_1', 'test_2'])
y = np.array(training_data['outcome'])

clf = neighbors.KNeighborsClassifier(3, weights='uniform')
trained_model = clf.fit(X, y)

trained_model.score(X, y)

# Create a new observation with the value of
# the first independent variable, 'test_1', as .4
# and the second independent variable, test_1', as .6
x_test = np.array([[.4, 0.6]])

# Apply the learner to the new, unclassified observation.
trained_model.predict(x_test)

trained_model.predict_proba(x_test)
 