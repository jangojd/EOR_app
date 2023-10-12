import pandas as pd
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import pandas as pd
import streamlit as st
import joblib
import pandas as pd
import pickle
#model = joblib.load('Rf_model.pkl')
# Load the saved Random Forest model
#model = joblib.load('Rf_model.pkl')
model=pickle.load(open("trained_model.sav","rb"))
# Create a function to make predictions
# Import necessary libraries
formations = ['S', 'Dolo', 'Carb', 'Sh', 'LS', 'US', 'Tripol', 'Cong', 'LS or Dolo', 'DoloorTripol', 'LSorDolo', 'Dolo or S', 'SS', 'Congl', 'SorLS-Dolo']

# Load the saved Random Forest model
#model = joblib.load('Rf_model.pkl')

# Create a function to make predictions
  # Cache this function to avoid recomputation
# Import necessary libraries
# Create a function to make predictions
@st.cache_data # Cache this function to avoid recomputation
def predict_eor_area(area_acres, porosity, permeability_md, depth_ft, gravity_api, viscosity_cp, temperature_f, sat_percent, formation):
    input_data = pd.DataFrame({
        'Area_acres': [area_acres],
        'Porosity_%': [porosity],
        'permeability_md': [permeability_md],
        'Depth_ft': [depth_ft],
        'Gravity (API)': [gravity_api],
        'Viscosity_cp ': [viscosity_cp],  # Use the column name with the space
        'Temprature_F': [temperature_f],
        'sat_%_100': [sat_percent],
        'Formation ': [formation]  # Use the column name with the space
    })

    prediction = model.predict(input_data)
    return prediction, input_data  # Return input_data as well

# Create the Streamlit web app
st.title('EOR Method Prediction App')

st.sidebar.header('User Input')

# Collect user input using widgets
area_acres = st.sidebar.number_input('Area (acres)', min_value=0.0, max_value=100000.0, value=5000.0, step=1.0)
porosity = st.sidebar.number_input('Porosity (%)', min_value=0.0, max_value=100.0, value=20.0, step=1.0)
permeability_md = st.sidebar.number_input('Permeability (md)', min_value=0.0, max_value=100000.0, value=100.0, step=1.0)
depth_ft = st.sidebar.number_input('Depth (ft)', min_value=0.0, max_value=1000000.0, value=5000.0, step=1.0)
gravity_api = st.sidebar.number_input('Gravity (API)', min_value=0.0, max_value=10000.0, value=30.0, step=1.0)
viscosity_cp = st.sidebar.number_input('Viscosity (cp)', min_value=0.0, max_value=100000.0, value=10.0, step=1.0)
temperature_f = st.sidebar.number_input('Temperature (°F)', min_value=0.0, max_value=30000.0, value=150.0, step=1.0)
sat_percent = st.sidebar.number_input('Saturation (%)', min_value=0.0, max_value=100.0, value=50.0, step=1.0)
formation = st.sidebar.selectbox('Formation', formations)  # Input field for Formation

# Make predictions when the user clicks the "Predict" button
if st.sidebar.button('Predict'):
    prediction, input_data = predict_eor_area(area_acres, porosity, permeability_md, depth_ft, gravity_api, viscosity_cp, temperature_f, sat_percent, formation)
    st.subheader('Predicted EOR Method:')
    
    # Increase the size of the result text
    st.markdown(f"<p style='font-size: 20px;'>Predicted EOR Method: {prediction[0]}</p>", unsafe_allow_html=True)
     
    # Create a DataFrame to hold class probabilities
    probability_df = pd.DataFrame({
        'EOR Method': model.classes_,
        'Probability': model.predict_proba(input_data)[0]
    })

    # Sort the DataFrame by probability in descending order
    probability_df = probability_df.sort_values(by='Probability', ascending=False)

    # Create an Altair bar chart
    st.subheader('Bar chart of EOR methods Probabilities:')
    chart = alt.Chart(probability_df).mark_bar().encode(
        x=alt.X('Probability:Q', axis=alt.Axis(format='%'), title='Probability'),
        y=alt.Y('EOR Method:N', sort='-x', title='EOR Method'),
        color=alt.Color('EOR Method:N', scale=alt.Scale(scheme='category20'), legend=None)
    ).properties(
        width=600,
        height=400
    )
    #Display the DataFrame of probabilities
    st.altair_chart(chart)
    st.subheader('Probabilities Of EOR Methods:')
    st.write(probability_df)    
    # Display the Altair chart
    
    # Display the authors and co-authors
st.sidebar.markdown('**Authors:**')
st.sidebar.markdown('Jawad Ali')
st.sidebar.markdown('**Co-Authors:**')
st.sidebar.markdown('Ali Akbar')
st.sidebar.markdown("Ali Zaheer")

# Add a link to the GitHub repository
st.sidebar.markdown('**GitHub Repository:**')
st.sidebar.markdown('[Link to GitHub Repository](https://github.com/jangojd/jangojd)')
   
