# Machine Learning Data PreProcessing Using Python Project 1:

• Data preprocessing is the crucial step in making a ML model.

• If there is no data preprocessing step on the data, your ML model won't work properly.

|Language Used: | Python |
|--- |--- |

|IDE Used: | PyCharm |
|--- |--- |

Pre-processing steps to do to prepare any dataset on which we will build ML model.

## Get the Dataset:

The first step is always to get the dataset and try to understand dataset. Try to figure out what all are independent variables and what are dependent variables. In ML, some independent variables are used to predict a dependent variable.

The name of the dataset I have taken is 'purchase_data.csv'.

Dataset contains four columns -> Country, Age, Salary, Purchased. Dataset has total 10 observations.

Dataset contains information of customers of some company and first three columns are information of a customer like country, age, salary and fourth column tell if the customer has purchased the product of the company or not.

In purchase_data.csv, first three columns i.e. Country, Age, Salary are independent variables whereas fourth column Purchased is dependent variable.

## Importing the Libraries:

Importing following three essential libraries

•	NumPy -> contains mathematical tools

•	Matplotlib -> to plot nice charts

•	Pandas -> to import the dataset and manage the dataset

## Import the Dataset:

For Python, you have to create the matrix of features (independent variables) and dependent variable vector.

Hence for purchase_data.csv, create the matrix of three independent variables and then create the dependent variable vector.

## Dealing with Missing Data:

Your dataset may contain missing data. In purchase_data.csv, we can see that there is one missing data in the age column for Spain and one missing data in salary column for Germany.

One way to deal with missing data is to remove the lines of the observation where there is some missing data but that can be dangerous because the data set may contain crucial information. So, it is quite dangerous to remove observation.

Another way to handle missing data is to take the mean of the columns. So, in age column where we have missing entry for Spain, we will replace this missing data by the mean of all the values in the column age.

In Python, Imputer class from Scikitlearn preprocessing library allows us to take care of missing data.

## Encode Categorical Data:

Your dataset may contain quantitative and qualitative variables. Quantitative variables contain numeric values whereas qualitative variables contain the categories or levels within the data.

ML models are based on mathematical equations, so it would cause some problem if we keep the text and use categorical variables in the equation because ML Model want only numbers in the equations. That's why we need to encode categorical variables.

In 'purchase_data.csv', Country and Purchased are two categorical variables because they simply contain categories.

• Country variable contains three categories - France, Spain, Germany

• Purchased variable contains two categories - yes, no

In Python, categorical data can be encoded using LabelEncoder class for scikit learn preprocessing. Hence, python will give levels to these categories and the order of those levels is not important. But we have to prevent ML equations from thinking one level is greater than other or vice versa. To prevent this, we use dummy variables. So, for example, for Country column, instead of having one column we will have 3 columns. This can be achieved with the help of OneHotEncoder class and ColumnTransformer class.

We don't need to use OneHotEncoder for Purchased variable which is a dependent variable. Since it is a dependent variable ML will know that it is a category and there is no order between the categories of Purchased variable.

## Split the Dataset into Training Data and Test Data:

ML is about a machine that is going to learn from the data to make predictions.

We need to split the dataset into training set and test set. Using training set, we build the machine learning model and using test set, we test the performance of this machine learning model.

We are building our ML model on training set by establishing some correlation between independent variables and dependent variables and once the ML model understands the correlation between independent variables and dependent variables. We will test if the ML model can apply the correlations you understood based on training set on test set.

In nutshell, we have to make two different datasets. The training set on which the machine learning model learns and test set on which we test if the ML model learned correctly the correlations.

In Python, model_selection class from scikitlearn library is used to split the dataset into training and test set.

## Feature Scaling:

In the dataset, the variables age and salary are not on the same scale because the Age is ranging from 27 to 50 and Salary is ranging 40k to 90k.

We need to have these variables in the same scale otherwise their distance (or Euclidean Distance) would be dominated by the variable with high range.

Standardization and normalization are two ways of doing feature scaling.

In Python, we use StandardScaler class from scikitlearn preprocessing library.
