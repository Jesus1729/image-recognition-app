import torch
import streamlit as st
import os
import uuid
from io import BytesIO
from PIL import Image
import cachetools
import ast

st.set_page_config(
    page_title="Image Recognition App in Snowflake",
    layout='wide'
)

st.header("Image Recognition App In Snowflake")
st.caption("Try uploading an image to get started with image recognition.")

@cachetools.cached(cache={})
def load_class_mapping(filename):
    with open(filename, "r") as f:
        return f.read()

# Load model and pre-process
@cachetools.cached(cache={})
def load_model():
    from torchvision import transforms
    import torch
    from mobilenetv3 import mobilenetv3_large

    model_file = "mobilenetv3-large-1cd25616.pth" 
    imgnet_class_mapping_file = "imagenet1000_clsidx_to_labels.txt"  

    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])

    cls_idx = load_class_mapping(imgnet_class_mapping_file)
    cls_idx = ast.literal_eval(cls_idx)

    model = mobilenetv3_large()
    model.load_state_dict(torch.load(model_file))

    model.eval().requires_grad_(False)

    return model, transform, cls_idx

def recognize_image(image_bytes_in_str):
    image_bytes = bytes.fromhex(image_bytes_in_str)
    model, transform, cls_idx = load_model()

    img = Image.open(BytesIO(image_bytes)).convert('RGB')
    img = transform(img).unsqueeze(0)

    logits = model(img)
    outp = torch.nn.functional.softmax(logits, dim=1)

    confidence, idx = torch.max(outp, 1)
    predicted_label = cls_idx[idx.item()]
    confidence = confidence.item()

    return predicted_label, confidence

uploaded_files = st.file_uploader("Choose image files", accept_multiple_files=True, label_visibility='hidden')
if uploaded_files is not None and len(uploaded_files) > 0:

    with st.spinner("Uploading images and generating predictions in real-time..."):

        predictions = []  

        for uploaded_file in uploaded_files:  
            bytes_data_in_hex = uploaded_file.getvalue().hex()

            file_base, file_extension = os.path.splitext(uploaded_file.name)
            file_name = f"{file_base}_{uuid.uuid4().hex}{file_extension}"

            predicted_label, confidence = recognize_image(bytes_data_in_hex)

            predictions.append((uploaded_file, predicted_label, confidence))

        for i, (uploaded_file, predicted_label, confidence) in enumerate(predictions):
            st.subheader(f"🔍 Predicción para la imagen {i+1}")
            if confidence is not None:
                st.write(f"**{predicted_label}** (Confianza: {confidence * 100:.2f}%)")
            else:
                st.write(f"**{predicted_label}** (Confianza no disponible)")

            st.subheader(f"🖼 Imagen {i+1}")
            st.image(uploaded_file, width=300)  
