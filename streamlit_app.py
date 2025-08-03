# streamlit_app.py

import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# --- Model Loading and Caching ---
# Use st.cache_resource to load the model only once

@st.cache_resource
def load_my_model():
  """Loads and caches the Keras model."""
  model_path = "Dog-breed-classification-data\models\20250729-14471753800432-inception_v3_acc_89.h5"
  model = tf.keras.models.load_model(model_path, compile=False)
  # The 'compile=False' argument is often helpful for inference-only models.
  return model

# Define the labels your model can predict
# IMPORTANT: Make sure the order matches your model's training output
UNIQUE_LABELS = ['affenpinscher', 'afghan_hound', 'african_hunting_dog', 'airedale',
       'american_staffordshire_terrier', 'appenzeller',
       'australian_terrier', 'basenji', 'basset', 'beagle',
       'bedlington_terrier', 'bernese_mountain_dog',
       'black-and-tan_coonhound', 'blenheim_spaniel', 'bloodhound',
       'bluetick', 'border_collie', 'border_terrier', 'borzoi',
       'boston_bull', 'bouvier_des_flandres', 'boxer',
       'brabancon_griffon', 'briard', 'brittany_spaniel', 'bull_mastiff',
       'cairn', 'cardigan', 'chesapeake_bay_retriever', 'chihuahua',
       'chow', 'clumber', 'cocker_spaniel', 'collie',
       'curly-coated_retriever', 'dandie_dinmont', 'dhole', 'dingo',
       'doberman', 'english_foxhound', 'english_setter',
       'english_springer', 'entlebucher', 'eskimo_dog',
       'flat-coated_retriever', 'french_bulldog', 'german_shepherd',
       'german_short-haired_pointer', 'giant_schnauzer',
       'golden_retriever', 'gordon_setter', 'great_dane',
       'great_pyrenees', 'greater_swiss_mountain_dog', 'groenendael',
       'ibizan_hound', 'irish_setter', 'irish_terrier',
       'irish_water_spaniel', 'irish_wolfhound', 'italian_greyhound',
       'japanese_spaniel', 'keeshond', 'kelpie', 'kerry_blue_terrier',
       'komondor', 'kuvasz', 'labrador_retriever', 'lakeland_terrier',
       'leonberg', 'lhasa', 'malamute', 'malinois', 'maltese_dog',
       'mexican_hairless', 'miniature_pinscher', 'miniature_poodle',
       'miniature_schnauzer', 'newfoundland', 'norfolk_terrier',
       'norwegian_elkhound', 'norwich_terrier', 'old_english_sheepdog',
       'otterhound', 'papillon', 'pekinese', 'pembroke', 'pomeranian',
       'pug', 'redbone', 'rhodesian_ridgeback', 'rottweiler',
       'saint_bernard', 'saluki', 'samoyed', 'schipperke',
       'scotch_terrier', 'scottish_deerhound', 'sealyham_terrier',
       'shetland_sheepdog', 'shih-tzu', 'siberian_husky', 'silky_terrier',
       'soft-coated_wheaten_terrier', 'staffordshire_bullterrier',
       'standard_poodle', 'standard_schnauzer', 'sussex_spaniel',
       'tibetan_mastiff', 'tibetan_terrier', 'toy_poodle', 'toy_terrier',
       'vizsla', 'walker_hound', 'weimaraner', 'welsh_springer_spaniel',
       'west_highland_white_terrier', 'whippet',
       'wire-haired_fox_terrier', 'yorkshire_terrier']

# --- Prediction Function ---
def get_preds(image_data, model):
    """Takes an image, preprocesses it, and returns the top 5 predictions."""
    # Preprocess the image to fit your model's input requirements
    # (e.g., resizing to 224x224, normalizing)
    image = image_data.convert('RGB')
    image = image.resize((299,299)) # Adjust size as per your model
    image = np.array(image)
    image = np.expand_dims(image, axis=0) # Add batch dimension
    image = tf.keras.applications.inception_v3.preprocess_input(image) # Preprocess for InceptionV3

    # Make prediction
    preds = model.predict(image)

    # Get top 5 predictions
    top_5_indexes = np.argsort(preds[0])[-5:][::-1]
    top_5_preds = [UNIQUE_LABELS[i] for i in top_5_indexes]
    top_5_conf = [preds[0][i] for i in top_5_indexes]
    return top_5_preds,top_5_conf

# --- Streamlit App UI ---
st.title("♻️ Trash Segregation Classifier")
st.write("Upload an image of trash, and the model will predict its category.")

# Load the model
model = load_my_model()

# Image uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write(uploaded_file)

    # Classify the image
    st.write("Classifying...")
    top_predictions,top_conf = get_preds(image, model)

    # Display the results
    st.success(f"**Top Prediction:** {top_predictions[0]} , **Confidence :** {top_conf[0]*100:.2f}%")
    st.write("**Other Possibilities:**")
    for pred in top_predictions[1:]:
        st.write(f"- {pred}")
