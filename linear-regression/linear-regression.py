"""Linear regression Boston dataset."""
# Load libraries
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
import warnings

# Suppress Warning
warnings.filterwarnings(action="ignore", module="scipy",
                        message="^internal gelsd")

# Load Boston data (X & Y)
boston = load_boston()
X = boston.data
y = boston.target

# Create linear regression
regr = LinearRegression()

# Fit the linear regression model
model = regr.fit(X, y)

# View the intercept
model.intercept_q

# View the feature coefficients
model.coef_
