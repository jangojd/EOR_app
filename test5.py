import pandas as pd

import matplotlib.pyplot as plt
from sklearn.compose import make_column_transformer
import seaborn as sns
import sklearn
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from category_encoders import OneHotEncoder
from category_encoders import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier

from IPython.display import VimeoVideo

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import make_scorer, f1_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import joblib
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

formations = ['S', 'Dolo', 'Carb', 'Sh', 'LS', 'US', 'Tripol', 'Cong', 'LS or Dolo', 'LSDolo', 'L', 'DoloorTripol', 'LSorDolo', 'Dolo or S', 'SS', 'Congl', 'SorLS-Dolo']

# Load the saved Random Forest model
#model = joblib.load('Rf_model.pkl')

# Create a function to make predictions
  # Cache this function to avoid recomputation
# Import necessary libraries
# Create a function to make predictions
@st.cache  # Cache this function to avoid recomputation
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
    return prediction

# Create the Streamlit web app
st.title('EOR Method Prediction App')

st.sidebar.header('User Input')

# Collect user input using widgets
area_acres = st.sidebar.number_input('Area (acres)', min_value=0.0, max_value=10000.0, value=5000.0, step=1.0)
porosity = st.sidebar.number_input('Porosity (%)', min_value=0.0, max_value=100.0, value=20.0, step=1.0)
permeability_md = st.sidebar.number_input('Permeability (md)', min_value=0.0, max_value=10000.0, value=100.0, step=1.0)
depth_ft = st.sidebar.number_input('Depth (ft)', min_value=0.0, max_value=100000.0, value=5000.0, step=1.0)
gravity_api = st.sidebar.number_input('Gravity (API)', min_value=0.0, max_value=100.0, value=30.0, step=1.0)
viscosity_cp = st.sidebar.number_input('Viscosity (cp)', min_value=0.0, max_value=10000.0, value=10.0, step=1.0)
temperature_f = st.sidebar.number_input('Temperature (°F)', min_value=0.0, max_value=300.0, value=150.0, step=1.0)
sat_percent = st.sidebar.number_input('Saturation (%)', min_value=0.0, max_value=100.0, value=50.0, step=1.0)
formation = st.sidebar.text_input('Formation')  # Input field for Formation

# Make predictions when the user clicks the "Predict" button
if st.sidebar.button('Predict'):
    prediction = predict_eor_area(area_acres, porosity, permeability_md, depth_ft, gravity_api, viscosity_cp, temperature_f, sat_percent, formation)
    st.write('Predicted EOR Method:', prediction[0])

# Add a link to your GitHub repository or provide additional information if needed
st.sidebar.markdown('[GitHub Repository](https://github.com/yourusername/your-repo)')

# You can customize the layout and design of your app further based on your preferences



