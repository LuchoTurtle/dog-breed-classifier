import streamlit as st
from PIL import Image
from classifier import detect_image

st.title("Dog Breed Classifier")
st.subheader("small proof of concept for Unilabs")
st.markdown(
"""
This little app is to provide a simple overview of the process of creating a simple, proof of concept classifier for dogs, specifying different breeds.

It is worth noting that the classifier only expects an image being only a dog, a cat or a human (it has detection for human faces). Anything else will be regarded as foreign.

We are using a pre-trained model because training a model long enough to be accurate would take too long.


You can check the notebook [here](https://colab.research.google.com/drive/1QuCZ5JTLu0N2Og8qWrxGgwn8CyzKU0oE), which explains the process of this small project.

""")


uploaded_file = st.file_uploader("Choose an image...", type=("jpg", "png", "jpeg"))

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Your uploaded image', use_column_width=True)
    st.write("")

    label = detect_image(uploaded_file)
    st.write(label)



