import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import numpy as np

#Load dataset from Excel file
df = pd.read_excel("Car_Purchasing_Data.xlsx")

#Show first 5 rows of data
print("First 5 rows:")
print(df.head())

#Show last 5 rows of data
print("\nLast 5 rows:")
print(df.tail())

#Show dataset shape
print("\nShape of dataset:")
print(df.shape)

#Show dataset information
print("\nDataset info:")
print(df.info())

#Check for missing values in each column
print("\nNull values in dataset:")
print(df.isnull().sum())

#Show relationships between variables
sns.pairplot(df)
plt.show()

#Create input dataset by removing unnecessary columns and target column
X = df.drop(columns=[
    'Customer Name',
    'Customer e-mail',
    'Country',
    'Car Purchase Amount'
])

#Create output dataset (target variable we want to predict)
y = df['Car Purchase Amount']

#Display input dataset
print("\nInput dataset (X):")
print(X.head())

#Display output dataset
print("\nOutput dataset (y):")
print(y.head())

#Show shapes of input and output datasets
print("\nShapes:")
print("X shape:", X.shape)
print("y shape:", y.shape)

#Create scalers to normalize data between 0 and 1
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

# Scale input features
X_scaled = scaler_X.fit_transform(X)

#Convert scaled input back to DataFrame
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

print("\nScaled Input dataset (X):")
print(X_scaled.head())

#Scale output target
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

#Convert scaled output back to DataFrame
y_scaled = pd.DataFrame(y_scaled, columns=['Car Purchase Amount'])

print("\nScaled Output dataset (y):")
print(y_scaled.head())

#Show scaled datasets again (confirmation)
print("\nFirst few rows of scaled input dataset (X_scaled):")
print(X_scaled.head())

print("\nFirst few rows of scaled output dataset (y_scaled):")
print(y_scaled.head())

#Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_scaled,
    test_size=0.2,
    random_state=42
)

#Show shapes of training and testing sets
print("\nShapes of datasets:")

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

#Show first few rows of training data
print("\nFirst few rows of X_train:")
print(X_train.head())

print("\nFirst few rows of y_train:")
print(y_train.head())

#Show first few rows of testing data
print("\nFirst few rows of X_test:")
print(X_test.head())

print("\nFirst few rows of y_test:")
print(y_test.head())


#It's AI TIME
print("\nTraining AI models...\n")

print("Initializing models...")

lr_model = LinearRegression()
print("Linear Regression created")

ridge_model = Ridge(alpha=1.0)
print("Ridge Regression created")

svr_model = SVR(kernel='rbf')
print("SVR model created")


#Time to train these b%#&*$s
print("\nTraining models...\n")

print("Training Linear Regression...")
lr_model.fit(X_train, y_train.values.ravel())
print("Linear Regression trained")

print("\nTraining Ridge Regression...")
ridge_model.fit(X_train, y_train.values.ravel())
print("Ridge Regression trained")

print("\nTraining SVR...")
svr_model.fit(X_train, y_train.values.ravel())
print("SVR trained")


#Mirror Mirror on the wall. Give me that data once for all
print("\nMaking predictions...\n")

lr_pred = lr_model.predict(X_test)
print("Linear Regression predictions done")

ridge_pred = ridge_model.predict(X_test)
print("Ridge Regression predictions done")

svr_pred = svr_model.predict(X_test)
print("SVR predictions done")

#Check 'em
print("\nCalculating model performance (RMSE)...\n")
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_pred))
svr_rmse = np.sqrt(mean_squared_error(y_test, svr_pred))

#Print 'em
print("\nModel Performance (RMSE):")

print("Linear Regression RMSE:", lr_rmse)
print("Ridge Regression RMSE:", ridge_rmse)
print("SVR RMSE:", svr_rmse)