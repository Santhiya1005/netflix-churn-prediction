Netflix Customer Churn Prediction

A Machine Learning web application that predicts whether a Netflix customer is likely to churn based on demographic and subscription details.
Built using Python, Scikit-learn, and Streamlit with a Netflix-style UI.

🚀 Live Demo

https://netflixchurnprediction.streamlit.app/

📌 Features

Predicts customer churn (Yes / No)

Displays churn probability

Netflix-themed modern UI

User-friendly input form

Real-time prediction

🧠 Machine Learning Model

Algorithm: Classification model (Logistic Regression / Random Forest)

Preprocessing:

Feature scaling using StandardScaler

One-hot encoding for categorical features

Evaluation: Accuracy & probability score

🛠 Tech Stack

Frontend: Streamlit

Backend: Python

Libraries:

pandas

numpy

scikit-learn

joblib

📂 Project Structure
NETFLIX-CHURN-PREDICTION/
│
├── app.py                  # Streamlit application
├── churn_model.pkl         # Trained ML model
├── scaler.pkl              # Feature scaler
├── feature_columns.pkl     # Model feature columns
├── requirements.txt        # Dependencies
└── README.md               # Project documentation

⚙️ How to Run Locally
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py

📊 Input Parameters

Age

Tenure (Months)

Monthly Charge

Satisfaction Score

Gender

Under 30

City (Optional)

📈 Output

Churn Prediction: Likely to Churn / Loyal Customer

Churn Probability: Percentage score

🎯 Use Case

This project helps businesses:

Identify customers at risk of leaving

Take preventive actions

Improve customer retention strategies
