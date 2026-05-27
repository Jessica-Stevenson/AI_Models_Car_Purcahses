import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import numpy as np
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


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

#Check duplicate values in dataset
print("\nDuplicate values in dataset:")
print(df.duplicated().sum())

#Basic statistical summary of dataset
print("\nStatistical summary of dataset:")
print(df.describe())


#Show relationships between variables
#sns.pairplot(df)
#plt.show()


#Create input dataset by removing unnecessary columns and target column
X = df.drop(columns=[
    'Customer Name',
    'Customer e-mail',
    'Country',
    'Car Purchase Amount'
])


#Identify irrelevant columns
print("\nRemoved columns:")
print("Customer Name, Customer e-mail, Country")

print("\nReason:")
print("These columns are identifiers or non-numeric and do not contribute to prediction")


#Create output dataset
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
ridge_pred = ridge_model.predict(X_test)
svr_pred = svr_model.predict(X_test)


#Check 'em
print("\nCalculating model performance (RMSE)...\n")

lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_pred))
svr_rmse = np.sqrt(mean_squared_error(y_test, svr_pred))

print("\nModel Performance (RMSE):")
print("Linear Regression RMSE:", lr_rmse)
print("Ridge Regression RMSE:", ridge_rmse)
print("SVR RMSE:", svr_rmse)


#Train Predictions predictions
lr_train_pred = lr_model.predict(X_train)
ridge_train_pred = ridge_model.predict(X_train)
svr_train_pred = svr_model.predict(X_train)


#Train RMSE
lr_train_rmse = np.sqrt(mean_squared_error(y_train, lr_train_pred))
ridge_train_rmse = np.sqrt(mean_squared_error(y_train, ridge_train_pred))
svr_train_rmse = np.sqrt(mean_squared_error(y_train, svr_train_pred))


#Does it overfit
print("\nOverfitting Check")

print("\nLinear Regression:")
print("Train RMSE:", lr_train_rmse)
print("Test RMSE :", lr_rmse)

print("\nRidge Regression:")
print("Train RMSE:", ridge_train_rmse)
print("Test RMSE :", ridge_rmse)

print("\nSVR:")
print("Train RMSE:", svr_train_rmse)
print("Test RMSE :", svr_rmse)


#Check extra model performance metrics
print("\nCalculating extra evaluation metrics...\n")


#Linear Regression metrics
lr_mae = mean_absolute_error(y_test, lr_pred)
lr_mse = mean_squared_error(y_test, lr_pred)
lr_r2 = r2_score(y_test, lr_pred)


#Ridge Regression metrics
ridge_mae = mean_absolute_error(y_test, ridge_pred)
ridge_mse = mean_squared_error(y_test, ridge_pred)
ridge_r2 = r2_score(y_test, ridge_pred)


#SVR metrics
svr_mae = mean_absolute_error(y_test, svr_pred)
svr_mse = mean_squared_error(y_test, svr_pred)
svr_r2 = r2_score(y_test, svr_pred)


print("\nModel Performance (All Metrics):")

print("\nLinear Regression:")
print("MAE:", lr_mae)
print("RMSE:", lr_rmse)
print("R2:", lr_r2)

print("\nRidge Regression:")
print("MAE:", ridge_mae)
print("RMSE:", ridge_rmse)
print("R2:", ridge_r2)

print("\nSVR:")
print("MAE:", svr_mae)
print("RMSE:", svr_rmse)
print("R2:", svr_r2)


#Create comparison table for all models
results = pd.DataFrame({
    "Model": ["Linear Regression", "Ridge Regression", "SVR"],
    "MAE": [lr_mae, ridge_mae, svr_mae],
    "RMSE": [lr_rmse, ridge_rmse, svr_rmse],
    "R2 Score": [lr_r2, ridge_r2, svr_r2]
})

print("\nModel Comparison Table:")
print(results)


#Create baseline model (average prediction)
print("\nCreating baseline model...\n")

baseline_prediction = y_train.mean()
baseline_pred = np.full(shape=len(y_test), fill_value=baseline_prediction)

baseline_mae = mean_absolute_error(y_test, baseline_pred)
baseline_mse = mean_squared_error(y_test, baseline_pred)
baseline_rmse = np.sqrt(baseline_mse)
baseline_r2 = r2_score(y_test, baseline_pred)


print("\nBaseline Model Performance:")
print("MAE:", baseline_mae)
print("RMSE:", baseline_rmse)
print("R2:", baseline_r2)


#Compare models with baseline
print("\nModel vs Baseline Comparison:")

print("Linear Regression better than baseline:", lr_rmse < baseline_rmse)
print("Ridge Regression better than baseline:", ridge_rmse < baseline_rmse)
print("SVR better than baseline:", svr_rmse < baseline_rmse)


#Select best model based on RMSE
print("\nSelecting best model...")

model_scores = {
    "Linear Regression": lr_rmse,
    "Ridge Regression": ridge_rmse,
    "SVR": svr_rmse
}

best_model_name = min(model_scores, key=model_scores.get)

if best_model_name == "Linear Regression":
    best_model = lr_model
elif best_model_name == "Ridge Regression":
    best_model = ridge_model
else:
    best_model = svr_model


#Save best model and scalers
joblib.dump(best_model, "best_model.pkl")
joblib.dump(scaler_X, "scaler_X.pkl")
joblib.dump(scaler_y, "scaler_y.pkl")


#UI

st.title("Car Purchase Prediction AI")

st.write("Enter customer details below to predict car purchase amount:")

gender = st.selectbox("Gender (0 = Male, 1 = Female)", [0, 1])

age = st.number_input("Age", min_value=0, max_value=120, value=30)

annual_salary = st.number_input("Annual Salary", min_value=0, value=50000)

credit_card_debt = st.number_input("Credit Card Debt", min_value=0, value=1000)

net_worth = st.number_input("Net Worth", min_value=0, value=10000)


if st.button("Predict"):

    #Input validation
    if annual_salary < 0:
        st.error("Annual Salary cannot be negative")
    elif credit_card_debt < 0:
        st.error("Credit Card Debt cannot be negative")
    elif net_worth < 0:
        st.error("Net Worth cannot be negative")
    else:

        input_data = pd.DataFrame([[
            gender,
            age,
            annual_salary,
            credit_card_debt,
            net_worth
        ]], columns=X.columns)

        input_scaled = scaler_X.transform(input_data)

        prediction_scaled = best_model.predict(input_scaled)

        prediction = scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1))

        st.success(f"Based on this dataset (likely USD market), predicted purchase is ${prediction[0][0]:,.2f}")


#Plots
st.title("Car Purchase Data Analysis")

st.subheader("Annual Salary vs Car Purchase Amount")
fig1 = plt.figure()
sns.scatterplot(data=df, x="Annual Salary", y="Car Purchase Amount")
st.pyplot(fig1)

st.subheader("Age vs Car Purchase Amount")
fig2 = plt.figure()
sns.scatterplot(data=df, x="Age", y="Car Purchase Amount")
st.pyplot(fig2)

st.subheader("Net Worth vs Car Purchase Amount")
fig3 = plt.figure()
sns.scatterplot(data=df, x="Net Worth", y="Car Purchase Amount")
st.pyplot(fig3)

st.subheader("Credit Card Debt vs Car Purchase Amount")
fig4 = plt.figure()
sns.scatterplot(data=df, x="Credit Card Debt", y="Car Purchase Amount")
st.pyplot(fig4)


#Model comparison chart
st.title("Model Performance Comparison")

models = ["Linear Regression", "Ridge Regression", "SVR"]
rmse_values = [lr_rmse, ridge_rmse, svr_rmse]

fig = plt.figure()
plt.bar(models, rmse_values)
plt.title("RMSE Comparison")
plt.ylabel("RMSE")
st.pyplot(fig)