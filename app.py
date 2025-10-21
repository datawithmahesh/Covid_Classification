#this is for covid classification(svm)
import streamlit as st
import pandas as pd
import pickle as pk
# st.image("https://pianalytix.com/wp-content/uploads/2020/12/Salary-Prediction-Model-using-ML-1024x427.jpg&quot")

st.write("COVID CLASSIFICATION MODEL")

# load model
load_model= pk.load(open('Covid_Classification.pickle','rb'))

#input data from user
cough_symptom = st.radio("Cough =",[True,False])
fever = st.radio("Fever =",[True,False])
sore_throat = st.radio("Sore_Stroat =",[True,False])
breadth_sortness = st.radio("Sortness of breadth =",[True,False])
headache = st.radio("Headache =",[True,False])
Known_contact = st.selectbox("known Contact",["Abroad", "Contact with confirmed", "Other"] )
age_above_60 = st.radio("Age_above_60 =",[True,False])
gender = st.radio("Sex = ", ["male", "female"])  #table ma 0 and 1 xa but 0 and 1 ra true false yeutai ho or male female rakhda tala dekhako xa

# model test paxi table ma jasto khal ko xa testai garne like true false 
# ani arko true false bhannu ra 0 and 1 bhannu yeutai ho 
# xuttai columns like bachelor master jasto bhako bhaye yesma ni if elif launu parthyo but yesma as type layera yeutai ma 0 and 1 xa or
# true false wala xa column ma ani tesko if condition launu parena 

if Known_contact == "Abroad":
     Known_contact = 0
    
elif Known_contact == "Contact with confirmed":
     Known_contact = 1
else:
     Known_contact = 2

#mapping
sex = {'male':True, 'female':False}



# Cough_symptoms	Fever	Sore_throat	Shortness_of_breath	Headache	Known_contact	Age_60_above	Sex

if st.button("predict"):
   df = pd.DataFrame({
       "Cough_symptoms":[cough_symptom],
       "Fever":[fever],
       "Sore_throat": [sore_throat],
       "Shortness_of_breath": [breadth_sortness],
       "Headache": [headache],
       "Known_contact":[Known_contact],
       "Age_60_above":[age_above_60],
       "Sex":[sex[gender]]
   })
   st.dataframe(df)
   result = load_model.predict(df)
   print(type(result)) # yo garda result numpy array ko form ma dekhinxa so teslai list ma convert gareni ani index 1 lai lageko

   if int(result.tolist()[0])== 0:
      st.write("Congratulations!!, you have negative result")
   else:
     st.write("we are sorry to inform you, you have covid positive, wish you the best health")



st.write("This model is done by Mahesh Thapa")