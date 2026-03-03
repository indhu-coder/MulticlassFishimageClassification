Finally the model has been chosen as Vision transformer for this project, which has highest accuracy of all the models experimented so far.

Model chosen:"Vision Transformer"

Pre-trained model: "google/vit-base-patch16-224"

Fine tuned the hyperparameter according to the fish dataset.

During training it gave the accuracy of 99.86% and for validation known images set the accuracy is 99.27%.

After model has been saved Train accuracy,Test accuracy and energy threshold was calculated as shown below:

Train Accuracy: 99.71%

Test Accuracy: 99.31%

Chosen Energy Threshold:  -2.5218210220336914

As for the Single image inference the codes are given below:    
    
    import streamlit as st
    import torch
    from transformers import  ViTForImageClassification, ViTImageProcessor
    from PIL import Image
    from App import load_model
    from Vision_transformers import processor
    from pyexpat import model
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    class_names = torch.load("models/class_names.pt")
    NUM_CLASSES = len(class_names)
    
    
    
    # -----------------------------
    # Streamlit UI
    # -----------------------------
    st.title("🐠 Fish Image Classification")
    st.subheader("Vision Transformer")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            if image.size[0] == 0 or image.size[1] == 0:
                st.error("Invalid image dimensions")
            else:
                st.image(image, caption="Uploaded Image")
        except Exception as e:
            st.error("Failed to load image")
            st.error(str(e))
    
        st.subheader("Prediction")
    
    # -----------------------------
    # Preprocessing & Prediction
    # -----------------------------
        model = ViTForImageClassification.from_pretrained("models/vit_11cls_v3")
        processor = ViTImageProcessor.from_pretrained("models/vit_11cls_v3")
        model.eval()
        ENERGY_T = -2.521821
        T = 3.5
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
                
            outputs = model(**inputs).logits
            confidence = torch.softmax(outputs, dim=1).max().item()
            energy = -torch.logsumexp(outputs / T, dim=1).item()
            pred = outputs.argmax(dim=1).item()
            if energy > ENERGY_T and confidence <= 0.85:
                st.write(f"Model: Vision Transformer | Unknown image detected | Confidence: {confidence * 100:.2f}")
            else:
                st.write(f"Model : Vision Transformer | Predicted: {class_names[pred]} | Confidence: {confidence * 100:.2f}")

