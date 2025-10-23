# COVID Classification Model
import streamlit as st
import pandas as pd
import pickle as pk

# Page config
st.set_page_config(
    page_title="COVID Classification Model",
    page_icon="🦠",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Load model
load_model = pk.load(open('Covid_Classification.pickle','rb'))

# --- Custom CSS ---
st.markdown("""
<style>
/* Background gradient */
body {
    background: linear-gradient(to right, #89f7fe, #66a6ff);
}

/* Title card */
.title-card {
    background: linear-gradient(to right, #ff416c, #ff4b2b);
    padding: 25px;
    border-radius: 15px;
    color: white;
    text-align: center;
    font-family: 'Arial Black', sans-serif;
    box-shadow: 0 8px 20px rgba(0,0,0,0.3);
    margin-bottom: 20px;
}

/* Input boxes */
.stRadio, .stSelectbox {
    background: #ffffffcc;
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 15px;
}

/* Predict button */
.stButton>button {
    background-color: #36D1DC;
    color: white;
    font-size: 18px;
    border-radius: 10px;
    padding: 10px 25px;
    transition: 0.3s;
}
.stButton>button:hover {
    background-color: #5B86E5;
}

/* Output card */
.output-card {
    background: linear-gradient(to right, #43e97b, #38f9d7);
    padding: 20px;
    border-radius: 15px;
    color: white;
    text-align: center;
    font-size: 22px;
    font-weight: bold;
    margin-top: 20px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.3);
}

/* Footer */
.footer {
    text-align:center;
    color: #555;
    margin-top: 40px;
    font-style: italic;
}
</style>
""", unsafe_allow_html=True)

# --- Title Section ---
st.markdown('<div class="title-card"><h1>🦠 COVID Classification Model</h1></div>', unsafe_allow_html=True)

# --- Input Section ---
cough_symptom = st.radio("Cough =", [True, False])
fever = st.radio("Fever =", [True, False])
sore_throat = st.radio("Sore Throat =", [True, False])
breadth_sortness = st.radio("Shortness of Breath =", [True, False])
headache = st.radio("Headache =", [True, False])
Known_contact = st.selectbox("Known Contact", ["Abroad", "Contact with confirmed", "Other"])
age_above_60 = st.radio("Age above 60 =", [True, False])
gender = st.radio("Sex =", ["male", "female"])

# Mapping Known_contact
Known_contact_map = {"Abroad":0, "Contact with confirmed":1, "Other":2}

# Mapping gender
sex = {'male':True, 'female':False}

# --- Prediction ---
if st.button("Predict 🟢"):
    df = pd.DataFrame({
        "Cough_symptoms":[cough_symptom],
        "Fever":[fever],
        "Sore_throat":[sore_throat],
        "Shortness_of_breath":[breadth_sortness],
        "Headache":[headache],
        "Known_contact":[Known_contact_map[Known_contact]],
        "Age_60_above":[age_above_60],
        "Sex":[sex[gender]]
    })
    
    st.dataframe(df)
    result = load_model.predict(df)
    
    if int(result.tolist()[0]) == 0:
        st.markdown('<div class="output-card">🎉 Congratulations! You have negative result.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="output-card">⚠️ You have COVID positive. Wish you the best health!</div>', unsafe_allow_html=True)

# --- Footer ---
st.markdown('<div class="footer">This model is done by <b>Mahesh Thapa</b></div>', unsafe_allow_html=True)






# #this is for covid classification(svm)
# import streamlit as st
# import pandas as pd
# import pickle as pk
# # st.image("https://pianalytix.com/wp-content/uploads/2020/12/Salary-Prediction-Model-using-ML-1024x427.jpg&quot")

# st.write("COVID CLASSIFICATION MODEL")

# # load model
# load_model= pk.load(open('Covid_Classification.pickle','rb'))

# #input data from user
# cough_symptom = st.radio("Cough =",[True,False])
# fever = st.radio("Fever =",[True,False])
# sore_throat = st.radio("Sore_Stroat =",[True,False])
# breadth_sortness = st.radio("Sortness of breadth =",[True,False])
# headache = st.radio("Headache =",[True,False])
# Known_contact = st.selectbox("known Contact",["Abroad", "Contact with confirmed", "Other"] )
# age_above_60 = st.radio("Age_above_60 =",[True,False])
# gender = st.radio("Sex = ", ["male", "female"])  #table ma 0 and 1 xa but 0 and 1 ra true false yeutai ho or male female rakhda tala dekhako xa

# # model test paxi table ma jasto khal ko xa testai garne like true false 
# # ani arko true false bhannu ra 0 and 1 bhannu yeutai ho 
# # xuttai columns like bachelor master jasto bhako bhaye yesma ni if elif launu parthyo but yesma as type layera yeutai ma 0 and 1 xa or
# # true false wala xa column ma ani tesko if condition launu parena 

# if Known_contact == "Abroad":
#      Known_contact = 0
    
# elif Known_contact == "Contact with confirmed":
#      Known_contact = 1
# else:
#      Known_contact = 2

# #mapping
# sex = {'male':True, 'female':False}



# # Cough_symptoms	Fever	Sore_throat	Shortness_of_breath	Headache	Known_contact	Age_60_above	Sex

# if st.button("predict"):
#    df = pd.DataFrame({
#        "Cough_symptoms":[cough_symptom],
#        "Fever":[fever],
#        "Sore_throat": [sore_throat],
#        "Shortness_of_breath": [breadth_sortness],
#        "Headache": [headache],
#        "Known_contact":[Known_contact],
#        "Age_60_above":[age_above_60],
#        "Sex":[sex[gender]]
#    })
#    st.dataframe(df)
#    result = load_model.predict(df)
#    print(type(result)) # yo garda result numpy array ko form ma dekhinxa so teslai list ma convert gareni ani index 1 lai lageko

#    if int(result.tolist()[0])== 0:
#       st.write("Congratulations!!, you have negative result")
#    else:
#      st.write("we are sorry to inform you, you have covid positive, wish you the best health")



# st.write("This model is done by Mahesh Thapa")