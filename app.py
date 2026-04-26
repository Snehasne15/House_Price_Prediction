# create environment for windows
# python -m venv myenv
# activate environment
# myenv\Scripts\activate
# pip install streamlit scikit-learn pandas seaborn numpy
# streamlit run app.py
import pickle
import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np

from sklearn.preprocessing import StandardScaler

#load model
model=pickle.load(open('lr_model.pkl','rb'))

#title for app
st.title("House Price Prediction App")

#create input features

squarefootage=st.number_input("Square_Footage",min_value=100,max_value=4999,value=600)
numbedrooms=st.number_input("Num_Bedrooms",min_value=1, max_value=5,value=2)
numbaths=st.number_input("Num_Bathrooms",min_value=1,max_value=3,value=2)
lotsize=st.number_input("Lot_Size",min_value=0.0,max_value=4.0,value=1.0)
garagesize=st.number_input("Garage_Size",min_value=0,max_value=2,value=1)
neighborhood=st.number_input("Neighborhood_Quality",min_value=1,max_value=10,value=2)
houseage=st.number_input("house_age",min_value=1,max_value=40,value=2)

#create dataframe
input_features = pd.DataFrame(
    {
        'Square_Footage':[squarefootage],
        'Num_Bedrooms':[numbedrooms],
        'Num_Bathrooms':[numbaths],
        'Lot_Size':[lotsize],
        'Garage_Size':[garagesize],
        'Neighborhood_Quality':[neighborhood],
        'house_age':[houseage] 
        }
)

#Standardscaler
scaler = StandardScaler()
input_features[['Square_Footage','Lot_Size','house_age']]=scaler.fit_transform(input_features[['Square_Footage','Lot_Size','house_age']])

#predictions

if st.button('Predict'):
  predictions= model.predict(input_features)[0]
  st.success(f"Predicted House Price: ₹ {predictions:,.2f}")
  