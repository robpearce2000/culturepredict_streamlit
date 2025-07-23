import streamlit as st
import h2o
from h2o.estimators.gbm import H2OGradientBoostingEstimator

st.title("CulturePredict AI")

# Start H2O (only once)
if "h2o_started" not in st.session_state:
    h2o.init()
    st.session_state["h2o_started"] = True

# Load the saved H2O model
model_path = "GBM_model_k_val_best"
model = h2o.load_model(model_path)

# Upload input data
uploaded_file = st.file_uploader("Upload your genome feature CSV")

if uploaded_file:
    import pandas as pd

    df = pd.read_csv(uploaded_file)
    h2o_df = h2o.H2OFrame(df)

    predictions = model.predict(h2o_df)

    st.subheader("Predicted Optimal Temperatures:")
    st.write(predictions.as_data_frame())
