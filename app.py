import pickle
import numpy as np
import streamlit as st
import sklearn
# Page Configuration
st.set_page_config(
    page_title="Sales Demand Prediction System", page_icon="📈", layout="centered"
)

st.title("🛒 Product Sales Demand Prediction & Classification System")
st.write(
    "Test your trained machine learning models interactively through this web"
    " app!"
)


# Load the serialized models
@st.cache_resource
def load_models():
  linear_model = pickle.load(open("linear_regression_model.pkl", "rb"))
  logistic_model = pickle.load(open("logistic_regression_model.pkl", "rb"))
  return linear_model, logistic_model


try:
  lin_model, log_model = load_models()

  # Input Section for User
  st.header("Enter Order Parameters")
  quantity_ordered = st.number_input(
      "Order Quantity", min_value=1, max_value=1000, value=30
  )
  price_each = st.number_input(
      "Price Each ($)", min_value=1.0, max_value=10000.0, value=50.0
  )

  # Prediction Button
  if st.button("Run Prediction"):
    input_data = np.array([[quantity_ordered, price_each]])

    # 1. Linear Regression Prediction (Revenue)
    predicted_revenue = lin_model.predict(input_data)[0]

    # 2. Logistic Regression Prediction (Demand Classification)
    demand_pred = log_model.predict(input_data)[0]

    # Display Results
    st.divider()
    st.subheader("📊 Prediction Results")

    col1, col2 = st.columns(2)
    with col1:
      st.metric(
          label="Predicted Sales Revenue", value=f"${predicted_revenue:,.2f}"
      )

    with col2:
      if str(demand_pred).lower() in ["1", "high", "true"]:
        st.success("Demand Status: **High Demand 🚀**")
      else:
        st.warning("Demand Status: **Low Demand 📉**")

except FileNotFoundError:
  st.error(
      "Model files (`.pkl`) not found in the directory! Please make sure your"
      " trained models are saved in the project folder."
  )