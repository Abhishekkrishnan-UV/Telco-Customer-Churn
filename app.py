import streamlit as st
import pandas as pd
import joblib

# Load saved model and scaler
svm_model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')

# CSS for background image with opacity, custom button styles, and horizontal radio buttons
page_bg_img = '''
<style>
.stApp {
  background-image: url("https://raw.githubusercontent.com/Abhishekkrishnan-UV/data/refs/heads/main/1000253423-01-01.jpeg");
  background-size: cover;
  background-position: center;
  background-attachment: fixed;
}
.stApp::before {
  content: "";
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background-color: rgba(255, 255, 255, 0.8); /* white overlay with 80% opacity */
  z-index: -1;
}

.stButton>button {
    background-color: transparent;
    border: 2px solid #0072ff;
    color: #0072ff;
    padding: 10px 20px;
    font-size: 16px;
    font-weight: bold;
    cursor: pointer;
    border-radius: 5px;
    transition: background-color 0.3s, color 0.3s;
}
.stButton>button:hover {
    background-color: #0072ff;
    color: white;
}

/* Horizontal Radio Buttons */
.stRadio {
    margin-top: 3px; /* Adjust the value to control the padding above the radio buttons */
}
.stRadio>div>label {
    display: inline-block;
    margin: 0 10px;
}

.stRadio>div>label>div {
    background-color: transparent !important;
    color: #0072ff !important;
    border: 2px solid #0072ff;
    border-radius: 5px;
    padding: 10px;
    font-size: 16px;
    font-weight: bold;
    cursor: pointer;
    transition: none !important;
}

/* Remove border for Collab Notebook radio button */
.stRadio > div > label:nth-child(3) > div {
    border: none !important;
    background-color: transparent !important;
    color: #0072ff !important;
    transition: none !important;
}

/* Remove border for About radio button */
.stRadio > div > label:nth-child(4) > div {
    border: none !important;
    background-color: transparent !important;
    color: #0072ff !important;
    transition: none !important;
}

/* Remove border and hover effect for Home and Model Details radio buttons */
.stRadio > div > label:nth-child(1) > div,
.stRadio > div > label:nth-child(2) > div {
    border: none !important;
    background-color: transparent !important;
    color: #0072ff !important;
    transition: none !important;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)

# Radio button for page navigation (Horizontal layout) with label visibility collapsed
selected_page = st.radio(
    "",
    options=["Home", "Colab Notebook","Model Details","About"],
    index=0,  # Default to 'Home'
    key="selected_page",
    horizontal=True,  # This is key to making the radio buttons horizontal
    label_visibility="collapsed"  # This hides the "Choose a Page" label
)

# Home page content
if selected_page == "Home":
    st.markdown("""
        <div style="text-align:center;margin-top:10px">
            <h2 style="color: black;font-size: 2.5rem">Welcome to the Telco Customer Churn Predictor!</h2>
            <p style="color: black;font-size: 1rem">This tool helps predict whether a customer will churn based on various factors such as their subscription details, support services, and monthly charges.</p>
        </div>
    """, unsafe_allow_html=True)

    st.sidebar.header("Enter Customer Data")
    tenure = st.sidebar.slider("Tenure (months)", 0, 100, 12)
    online_security = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
    monthly_charges = st.sidebar.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=50.0)

    input_data = pd.DataFrame({
        'tenure': [tenure],
        'OnlineSecurity': [online_security],
        'OnlineBackup': [online_backup],
        'DeviceProtection': [device_protection],
        'TechSupport': [tech_support],
        'Contract': [contract],
        'PaperlessBilling': [paperless_billing],
        'MonthlyCharges': [monthly_charges]
    })

    input_data['OnlineSecurity'] = input_data['OnlineSecurity'].map({'Yes': 2, 'No': 0, 'No internet service': 1})
    input_data['OnlineBackup'] = input_data['OnlineBackup'].map({'Yes': 2, 'No': 0, 'No internet service': 1})
    input_data['DeviceProtection'] = input_data['DeviceProtection'].map({'Yes': 2, 'No': 0, 'No internet service': 1})
    input_data['TechSupport'] = input_data['TechSupport'].map({'Yes': 2, 'No': 0, 'No internet service': 1})
    input_data['Contract'] = input_data['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})
    input_data['PaperlessBilling'] = input_data['PaperlessBilling'].map({'Yes': 2, 'No': 0})

    input_data_scaled = scaler.transform(input_data)

    # Create a transparent "Predict Churn" button
    if st.button("Predict Churn"):
        churn_prediction = svm_model.predict(input_data_scaled)
        result = '"𝚃𝚑𝚎 𝚌𝚞𝚜𝚝𝚘𝚖𝚎𝚛 𝚠𝚒𝚕𝚕 𝚙𝚛𝚘𝚋𝚊𝚋𝚕𝚢 𝚌𝚑𝚞𝚛𝚗"' if churn_prediction == 1 else '"𝚃𝚑𝚎 𝚌𝚞𝚜𝚝𝚘𝚖𝚎𝚛 𝚠𝚒𝚕𝚕 𝚗𝚘𝚝 𝚌𝚑𝚞𝚛𝚗"'

        color = "red" if churn_prediction[0] == 1 else "green"
        st.markdown(
            f"<h2 style='color: {color}; text-align:center;'>{result}</h2>",
            unsafe_allow_html=True)

# Collab Notebook page content
elif selected_page == "Colab Notebook":
    st.markdown("<h2 style='color: black;font-size: 2rem'>Google Colab Notebook</h2>", unsafe_allow_html=True)
    st.markdown("""
        <p style="color: black;">Click the link below to open the Google Colab notebook:</p>
        <a href="https://colab.research.google.com/drive/1DgTgyJ94bEYrKYQq_enqO0XforYxnKSa?usp=sharing" 
           target="_blank" style="color: blue; text-decoration: underline;">Open Colab Notebook</a>
    """, unsafe_allow_html=True)



# Model Details page content

elif selected_page == "Model Details":
    st.markdown("""
               <div style="text-align:left;margin-top:10px">
                   <h2 style="color: black;font-size: 2rem;">Model Details and Evaluation</h2>

               </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <h2 style='color: black; font-size: 1.5rem;'>Model Used (Random Forest Classifier)</h2>
        <p style="color:black; margin-bottom: 10px;">Random Forest Classifier is used for classification. It  works by creating multiple decision trees during training and aggregating their predictions to improve accuracy and reduce overfitting.</p>

        <h2 style='color: black; font-size: 1.5rem;'>Encoder Used (Label Encoder)</h2>
        <p style="color:black; margin-bottom: 10px;">Label Encoder is used to convert categorical labels into numerical values, typically used when the categories have an ordinal relationship.</p>

        <h2 style='color: black; font-size: 1.5rem;'>Scaler Used (MinMaxScaler)</h2>
        <p style="color:black; margin-bottom: 10px;">MinMaxScaler is used to scale the data so that all features are within the range of 0 to 1. This is useful when features have different units or scales.</p>
    """, unsafe_allow_html=True)
    y_true = joblib.load('ytrue.pkl')
    y_pred = joblib.load('ypred.pkl')



    # confusion matrix insights

    st.image("https://raw.githubusercontent.com/Abhishekkrishnan-UV/data/refs/heads/main/cm2.png",
             width=500)


    st.markdown("<h2 style='color: black;font-size: 1.3rem'>Insights from the Confusion Matrix", unsafe_allow_html=True)


    st.markdown("""
    <div style="color:black;">

    - **True Positives :**  
      Customers correctly predicted as churning. High value indicates the model is good at identifying churners.

    - **True Negatives :**  
      Customers correctly predicted as *not* churning. High value indicates the model accurately identifies loyal customers.

    - **False Positives :**  
      Customers incorrectly predicted as churning.
      These are loyal customers wrongly flagged as churners. High false positives suggest the model is too sensitive and may lead to unnecessary interventions.

    - **False Negatives :**  
      Customers incorrectly predicted as *not* churning.  
      These are churned customers missed by the model. High false negatives are problematic as they represent lost customers the company failed to retain.
    </div>
    """, unsafe_allow_html=True)





# roc curve
    st.markdown("<h3 style='color: black;font-size: 1.5rem'>Receiver Operating Characteristic curve</h3>", unsafe_allow_html=True)
    st.image("https://raw.githubusercontent.com/Abhishekkrishnan-UV/data/refs/heads/main/roc2.png",
             width=500)


    #insights
    st.markdown("<h3 style='color:black;font-size:1.3rem;'>Insights from the ROC Curve", unsafe_allow_html=True)


    st.markdown("""
       <div style="color:black;">

       - **AUC Score:**  
         0.83, indicating good ability to distinguish between churned and non-churned customers.
         
       - **Model Performance:**  
         ROC curve shows the trade-off between true positive rate (sensitivity) and false positive rate (1-specificity) at different thresholds.
         
       - **Threshold Selection:**  
         Different points on the ROC curve represent different classification thresholds, with the optimal threshold balancing false positives and false negatives.
         
        </div>
       """, unsafe_allow_html=True)

#Classification report

    st.markdown("<h3 style='color: black;font-size: 1.5rem'>Classification Report</h3>", unsafe_allow_html=True)
    st.image("https://raw.githubusercontent.com/Abhishekkrishnan-UV/data/refs/heads/main/classificationReport.jpg",width=500)

    st.markdown("""
        <div style="color:black;">
        
        - **Precision**: Measures the accuracy of positive predictions. The proportion of true positives out of all predicted positives.
        
        - **Recall**: Measures the ability of the model to capture all positive instances. The proportion of true positives out of all actual positives.
        
        - **F1-Score**: The harmonic mean of precision and recall. It provides a balance between both metrics, especially when class distribution is uneven.
        
        - **Support**: The number of actual occurrences of each class in the dataset, helping to understand how many samples belong to each class.
        
        </div>
    """, unsafe_allow_html=True)

elif selected_page == "About":
    st.markdown("""
           <div style="text-align:left;margin-top:10px">
               <h2 style="color: black;font-size: 2rem;">About Telco Customer Churn</h2>
           </div>
                """,unsafe_allow_html=True)
    st.markdown("""
    <div style="float: right; margin: 20px; width: 50%; text-align: center;">
        <img src="https://raw.githubusercontent.com/Abhishekkrishnan-UV/data/refs/heads/main/churn%20(2).png" 
        alt="Flowchart of Churn Prediction Workflow" style="max-width: 100%; height: auto;">
        <p style="font-style: italic; font-size: 0.9rem; color:black;">Churn flow diagram</p>
    </div>
    <p style="color: black;font-size: 1rem">Customer churn refers to the loss of clients or customers to a competitor or due to dissatisfaction with a service. In the telecommunications (Telco) industry, churn is a critical metric for evaluating customer retention and the overall health of a business. A customer churn prediction model uses historical data to predict the likelihood that a customer will cancel or stop using the service.</p>
        <div style="text-align:left;margin-top:10px">
            <p style="color: black;font-size: 1rem">In this project, we focus on predicting customer churn for a telecommunications company. The model leverages various customer features such as tenure, contract type, tech support, and monthly charges, among others, to identify patterns and predict if a customer is likely to churn.</p>
            <h3 style="color: black;font-size: 1.5rem">Why Is Churn Important in Telco?</h3>
            <p style="color: black;font-size: 1rem">Churn in the telecom industry represents a loss in revenue and can indicate poor customer satisfaction or better offerings from competitors. By predicting which customers are likely to churn, companies can take proactive measures to retain them, such as offering discounts, improving customer service, or creating targeted marketing campaigns.</p>
            <h3 style="color: black;font-size: 1.5rem">How This Model Works</h3>
            <p style="color: black;font-size: 1rem">The model uses machine learning techniques like Support Vector Machines (SVM) to classify customers into 'likely to churn' or 'not likely to churn' categories. The data used for training the model includes customer demographics, service details, and usage patterns. The SVM algorithm is trained on this data to learn the decision boundary that separates the two classes.</p>
            <h3 style="color: black;font-size: 1.5rem">Features used in this Model</h3>
            <ul style="color: black;font-size: 1rem; text-align: left;">
                <li><b>Tenure:</b> The length of time a customer has been with the service.</li>
                <li><b>Contract Type:</b> Whether the customer has a month-to-month contract or a longer-term contract.</li>
                <li><b>Tech Support:</b> Whether the customer has opted for technical support services.</li>
                <li><b>Monthly Charges:</b> The amount the customer is billed monthly for the services.</li>
                <li><b>Online Security:</b> Whether the customer has subscribed to online security services.</li>
                <li><b>Paperless Billing:</b> Whether the customer uses paperless billing.</li>
            </ul>
            <p style="color: black;font-size: 1rem">This model can help telecommunications companies predict which customers may churn, allowing for timely interventions and enhanced customer retention strategies.</p>
        </div>
    """, unsafe_allow_html=True)
