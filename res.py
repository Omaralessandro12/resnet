import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from torchvision.models import resnet50
from torchvision import models

def load_model(model_path):
    model = models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(2048, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch

def predict(image, model):
    with torch.no_grad():
        input_tensor = preprocess_image(image)
        output = model(input_tensor)
        return output

def main():
    st.title("Aplicación de Detección de Objetos con ResNet-50")

    uploaded_file = st.file_uploader("Cargar imagen", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen cargada", use_column_width=True)

        model_path = 'ruta/al/archivo/modelo.pt'  # Reemplazar con la ruta correcta a tu archivo .pt
        model = load_model(model_path)

        prediction = predict(image, model)

        st.write("Resultados de la predicción:")
        st.json(prediction.tolist())

if __name__ == "__main__":
    main()
