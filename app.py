import streamlit as st
from PIL import Image
import numpy as np

st.set_page_config(page_title="Cat vs Dog Classifier")

st.title("🐱 Cat vs Dog Classifier 🐶")
st.write("Upload a cat or dog image and I'll tell you what it is!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Model loading abhi disable hai. TensorFlow hata diya tha na.")
    st.success("Prediction: Ye feature next update mein add hoga ✅")
