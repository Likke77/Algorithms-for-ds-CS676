import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ml_trainer import train_model, is_continuous

st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("AutoML Model Builder")

st.write("Upload your dataset and let the app do the EDA and model training!")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("1. Data Preview:")
    st.dataframe(data)

    # Basic EDA
    st.subheader("2. Basic Data Overview:")
    st.write(f"**Shape of data:** {data.shape[0]} rows, {data.shape[1]} columns")
    st.write("**Column types:**")
    st.write(data.dtypes)
    st.write("**Summary statistics:**")
    st.write(data.describe(include='all'))

    # Missing values
    st.subheader("3. Missing Values:")
    missing = data.isnull().sum()
    st.write(missing[missing > 0])

    # Correlation heatmap (only numerical)
    if data.select_dtypes(include=['number']).shape[1] > 1:
        st.subheader("4. Correlation Heatmap:")
        st.set_option('deprecation.showPyplotGlobalUse', False)
        fig = plt.figure(figsize=(8, 6))
        sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        st.pyplot(fig)

    # Target column selection
    target_column = st.selectbox("5. Select the target column", data.columns)

    if st.button("6. Train Model"):
        # Prepare data
        st.subheader("7. Data Preparation:")

        # Fill missing numerical columns
        num_cols = data.select_dtypes(include=['float64', 'int64']).columns
        data[num_cols] = data[num_cols].fillna(data[num_cols].median())
        st.write("Filled missing numeric values with median.")

        # One-hot encode categorical columns
        cat_cols = data.select_dtypes(include=['object', 'category']).columns.difference([target_column])
        data = pd.get_dummies(data, columns=cat_cols, drop_first=True)
        st.write("Encoded categorical columns using one-hot encoding.")

        y = data[target_column]
        if is_continuous(y):
            st.info("Detected continuous target. Suggested model: Linear Regression")
            model_type = "linear_regression"
        else:
            st.info("Detected categorical target. Suggested model: Logistic Regression")
            model_type = "logistic_regression"

        payload = {
            "data": data.to_dict(orient="records"),
            "target_column": target_column,
            "model_type": model_type
        }

        result = train_model(payload)
        st.success(f"Model Trained: {result['model_type']}")
        st.metric(label=result["metric_name"], value=round(result["score"], 4))
