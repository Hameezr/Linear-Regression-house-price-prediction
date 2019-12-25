import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sns as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# For reading the data
df = pd.read_csv('C:\\Files\\FAST NUCES\\7th Semester\\Artificial Intelligence\\A5\\Regression-house_data.csv')

# To remove the infinite and NaN values
df.replace('null',np.NaN, inplace=True)
df.replace(r'^\s*$', np.NaN, regex=True, inplace=True)
df.fillna(value=df.mean(), inplace=True)

# Feature engineering using a heat map
X_loc = df.iloc[:, 1:21]
cor = X_loc.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

# Using ID and price
X = pd.DataFrame(np.c_[df['id'], df['price']], columns=['id', 'price'])
Y = df['price']

# Splitting training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
print("X train data")
print(X_train.head())
print(X_train.shape)

print("X test data")
print(X_test.head())
print(X_test.shape)

# Applying Linear regression
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)

# Predicting the best values
y_train_predict = lin_model.predict(X_train)

# Calculating Root Mean Square Error
rmse = (np.sqrt(mean_squared_error(y_train, y_train_predict)))
r2 = r2_score(y_train, y_train_predict)

print("RESULTS MODEL FOR TRAINING DATA!!")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))

# model evaluation for testing set
y_test_predict = lin_model.predict(X_test)
rmse = (np.sqrt(mean_squared_error(y_test, y_test_predict)))
r2 = r2_score(y_test, y_test_predict)

print("RESULTS MODEL FOR TESTING DATA")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))

