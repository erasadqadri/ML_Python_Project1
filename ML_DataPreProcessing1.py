#Author: Asad Qadri
# ML Data Preprocessing - Python Project 1

# Importing essential libraries:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing Dataset:
df = pd.read_csv("purchase_data.csv")

#Creating matrix of independent variables (Country, Age and Salary)
x = df.iloc[:,:-1].values

#Creating matrix of independent variables (Purchased)
y = df.iloc[:,3].values

# Taking care of missing data
from sklearn.impute import SimpleImputer

# Creating the object of Imputer class
impute = SimpleImputer(missing_values = np.nan , strategy = 'mean')

# fit imputer object to data x (Matrix of feature x)
imputer = impute.fit(x[:, 1:3])

# Replace the missing data of column by mean
x[:, 1:3] = imputer.transform(x[:, 1:3])

# Encoding categorical data
# Importing LabelEncoder, OneHotEncoder and ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Creating the object of LabelEncoder class
labelencoder_x = LabelEncoder()
# fit labelencoder_x object to first coulmn Country of matrix x
x[:, 0] = labelencoder_x.fit_transform(x[:, 0])

#ColumnTransform and Fit Country Column
ct_x = ColumnTransformer(transformers=[('encode',OneHotEncoder(),[0])], remainder='passthrough')
x = ct_x.fit_transform(x)

# Creating the object of LabelEncoder class
labelencoder_y = LabelEncoder()
# fit labelencoder_y object to last coulmn Purchased, we will get encoded vector
y = labelencoder_y.fit_transform(y)

# Splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

# Creating the object of StandardScaler
sc_X = StandardScaler()

# fit and transform training set
X_train = sc_X.fit_transform(X_train)

# transform test set
X_test = sc_X.transform(X_test)