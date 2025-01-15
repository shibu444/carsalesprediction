import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error, r2_score

# Title of the Streamlit App
st.title("Model Evaluation App")

# Section for uploading the model
st.header("Upload Model")
uploaded_model_file = st.file_uploader("Upload your trained model (.py or .pkl)", type=["py", "pkl"])

if uploaded_model_file is not None:
    st.success("Model uploaded successfully!")

    # Load the model (for .py files, use exec or a safer method)
    if uploaded_model_file.name.endswith(".pkl"):
        model = pickle.load(uploaded_model_file)
    else:
        exec(uploaded_model_file.read().decode('utf-8'))  # Unsafe, ensure trusted files only

# Section for uploading train data
st.header("Upload Training Data")
train_file = st.file_uploader("Upload your training data (CSV file)", type=["csv"])

if train_file is not None:
    train_data = pd.read_csv(train_file)
    st.success("Training data uploaded successfully!")
    st.write("Training Data Preview:")
    st.dataframe(train_data.head())

    if 'cnt' in train_data.columns:
        y_train = train_data['cnt']
        X_train = train_data.drop(columns=['cnt'])

        if 'regmodel_new' in locals() or 'regmodel_new' in globals():
            train_pred = regmodel_new.predict(X_train)
            train_r2 = r2_score(y_train, train_pred)
            train_rmse = mean_squared_error(y_train, train_pred, squared=False)

            st.subheader("Training Data Evaluation")
            st.write(f"R²: {train_r2:.4f}")
            st.write(f"RMSE: {train_rmse:.4f}")

# Section for uploading test data
st.header("Upload Test Data")
test_file = st.file_uploader("Upload your test data (CSV file)", type=["csv"])

if test_file is not None:
    test_data = pd.read_csv(test_file)
    st.success("Test data uploaded successfully!")
    st.write("Test Data Preview:")
    st.dataframe(test_data.head())

    if 'cnt' in test_data.columns:
        y_test = test_data['cnt']
        X_test = test_data.drop(columns=['cnt'])

        if 'regmodel_new' in locals() or 'regmodel_new' in globals():
            test_pred = regmodel_new.predict(X_test)
            test_rmse = mean_squared_error(y_test, test_pred, squared=False)

            st.subheader("Test Data Evaluation")
            st.write(f"RMSE: {test_rmse:.4f}")

# Note: Ensure the uploaded Python file or pickled model is trusted and compatible.
