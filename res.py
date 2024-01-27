import streamlit as st
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
import numpy as np

# Cargar el modelo ResNet50 preentrenado
model = ResNet50(weights='imagenet')

def predict_image(img_path):
    # Cargar la imagen y preprocesarla para hacerla compatible con ResNet50
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Realizar la predicción
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=3)[0]

    return decoded_predictions

st.title("Aplicación de Detección de Objetos con ResNet50")

uploaded_file = st.file_uploader("Cargar una imagen...", type="jpg")

if uploaded_file is not None:
    st.image(uploaded_file, caption='Imagen de entrada.', use_column_width=True)
    st.write("")
    st.write("Clasificando...")

    # Hacer la predicción
    predictions = predict_image(uploaded_file)
    
    st.write(f"Predicciones (Top 3):")
    for i, (imagenet_id, label, score) in enumerate(predictions):
        st.write(f"{i + 1}: {label} ({score:.2f})")

