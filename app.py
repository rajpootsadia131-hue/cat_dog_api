import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

st.set_page_config(page_title="Cat vs Dog Classifier")
st.title("🐱 Cat vs Dog Classifier 🐶")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('model.h5') # apni model file ka naam
    return model

model = load_model()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((150, 150)) # apne model ke hisaab se size
    st.image(image, caption='Uploaded Image', use_column_width=True)

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    if prediction[0][0] > 0.5:
        st.success("Prediction: Dog 🐶")
    else:
        st.success("Prediction: Cat 🐱")

    st.write(f"Confidence: {prediction[0][0]:.2%}")
