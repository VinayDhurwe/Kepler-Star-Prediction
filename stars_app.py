import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from PIL import Image



hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 



st.write(
    """
    # Nasa Stars Classfication database

    """
    
)



st.write(
    """
   
## Purpose:
The purpose of making the project is to prove that the stars follows a certain graph in the celestial Space ,
specifically called **Hertzsprung-Russell Diagram** or simply **HR-Diagram**
so that we can classify stars by plotting its features based on that graph.

    """
    
)


st.write(
    """
    
    ## Dataset Info:

    This is a dataset consisting of several features of stars.

    Some of them are:

    1. Absolute Temperature (in K)
    2. Relative Luminosity (L/Lo)
    3. Relative Radius (R/Ro)
    4. Absolute Magnitude (Mv)
    5. Star Color (white,Red,Blue,Yellow,yellow-orange etc)
    6. Spectral Class (O,B,A,F,G,K,,M)
    7. Star Type **(Red Dwarf, Brown Dwarf, White Dwarf, Main Sequence , SuperGiants, HyperGiants)**
    8. Lo = 3.828 x 10^26 Watts (Avg Luminosity of Sun)
    9. sRo = 6.9551 x 10^8 m (Avg Radius of Sun)
    """
    
)

df = pd.read_csv("cleaned_stars.csv", sep="\t")
df.drop("Unnamed: 0", inplace=True, axis = 1)
df = df.head(10)

st.write("""
### Example database
""")

st.dataframe(df)



image1 = Image.open('stars type.jpg')

st.image(image1, caption='Tempreature of Stars')

image2 = Image.open('stars type2.jpg')

st.image(image2, caption='Lumonosity  of stars')

image3 = Image.open('stars type3.jpg')

st.image(image3, caption='Color by Lumonosity and Absolute Magnitude')






st.sidebar.header(
"""User Input Feature"""

)

st.sidebar.markdown("""
[Example CSV input file](https://github.com/VinayDhurwe/Kepler-Star-Prediction/blob/main/Stars.csv)
""")

uploaded_file = st.sidebar.file_uploader("Upload your input csv file here", type=['csv'])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        temp = st.sidebar.slider("Tempreature",1900,40000,10000)
        L = st.sidebar.slider("Lumonocity",0.00008,850000.0,107000.0)
        R = st.sidebar.slider("Rotation",0.00840,1950.0,238.0)
        A_M = st.sidebar.slider("Absolute Magnitude",-12,21,5)
        Color = st.sidebar.selectbox('Color',('Red','White','Blue','Yellow','Yellow White','Yellowish White','Pale Yellow Orange','Orange','Yellowish'))
        spectral = st.sidebar.selectbox('Spectral Class',('M','B','A','F','O','K','G'))
        data = {
            'Temperature' : temp,
            'L' : L,
            'R' : R,
            'A_M' : A_M,
            'Color' : Color,
            'Spectral_Class' : spectral
        }
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

#print(input_df)



stars_raw = pd.read_csv("cleaned_stars.csv", sep="\t")
df = pd.concat([input_df,stars_raw], axis = 0)
df.drop("Unnamed: 0", inplace=True, axis = 1)

print("this is the concated dattaframe : \n ",df.head(10))




le = LabelEncoder()
df["Color"] = le.fit_transform(df["Color"])
df["Spectral_Class"] = le.fit_transform(df["Spectral_Class"])

scaler = StandardScaler()
model = scaler.fit(df)
scaled_data = model.transform(df)
print(scaled_data)

model_rf = pickle.load(open('finalized_model_RandomForest.pkl', 'rb'))

df = scaled_data[:1]

print(df)

# Apply model to make predictions
prediction = model_rf.predict(df)
prediction_proba = model_rf.predict_proba(df)
print(prediction,'\n',prediction_proba)


st.subheader('Prediction')
star_type = np.array(['It is of Category Red Drawf','It is of Category Brown Drawf','It is of Category White Drawf','It is of Category Main Sequence','It is of Category Super Gaint','It is of Category Hyper Gaints'])
star_image =np.array(["red drawf.jpg","brown drawf.jpg","white drawf.jpg","main sequence.jfif","super giant.jpg","hyper giant.jfif"])
st.write(star_type[prediction])
print(type(star_image[prediction]))
img = star_image[prediction]
img = img.tolist()
img = str(img[0])
print(img)
star_img = Image.open(img)
st.image(star_img)

st.subheader('Prediction Probability')
st.write(prediction_proba)



