import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.cm as cm

# --- Model Loading and Caching ---
@st.cache_resource
def load_model():
    """Loads the pre-trained EfficientNet model."""
    try:
        model_path = r""D:\internship\project\Pneumonia-Detection-ML\pneumonia_detector_model.h5""
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        st.stop()

# --- Grad-CAM Functions ---
def get_last_conv_layer(model):
    """Finds the last convolutional layer in the model."""
    for layer in reversed(model.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
            return layer.name
    return None

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy()

def display_gradcam(original_img, heatmap, alpha=0.4):
    """Overlays the heatmap on the original image."""
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = Image.fromarray((jet_colors[heatmap] * 255).astype(np.uint8))
    jet_heatmap = jet_heatmap.resize(original_img.size)

    original_rgb = original_img.convert("RGB")
    superimposed_img = np.array(jet_heatmap) * alpha + np.array(original_rgb)
    superimposed_img = Image.fromarray(superimposed_img.astype(np.uint8))
    
    return superimposed_img

# --- Streamlit UI ---
st.set_page_config(page_title="Pneumonia Detection", page_icon="🩻")
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "Pneumonia Detection", "About"])

if app_mode == "Home":
    st.header("Pneumonia Detection Dashboard")
    st.image(
        "https://images.unsplash.com/photo-1579895059298-0c6d9d1b09b9?q=80&w=2070&auto=format&fit=crop",
        use_container_width=True,
        caption="An illustrative medical image dashboard"
    )
    st.markdown("""
    Welcome to the **Pneumonia Detection Dashboard**! This application uses a deep learning model to analyze chest X-ray images.
    
    **How to use:**
    - Navigate to the **Pneumonia Detection** page in the sidebar.
    - Upload a chest X-ray image.
    - The model will classify the image as **NORMAL** or **PNEUMONIA** and provide a visual explanation using Grad-CAM.
    """)

elif app_mode == "Pneumonia Detection":
    st.title("🩻 Chest X-Ray Pneumonia Detection")
    st.markdown("A fine-tuned **EfficientNetB0** model for classifying chest X-ray images.")

    with st.spinner("Loading the deep learning model..."):
        model = load_model()
        last_conv_layer_name = get_last_conv_layer(model)
        if not last_conv_layer_name:
            st.error("Could not find the last convolutional layer. Check the model architecture.")
            st.stop()
        st.success("Model loaded successfully!")

    st.subheader("Upload a Chest X-ray Image")
    uploaded_file = st.file_uploader(
        "Choose a PNG or JPG file...",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file:
        try:
            pil_image = Image.open(uploaded_file).convert("RGB")  # ensure RGB
        except Exception as e:
            st.error(f"Error processing the image: {e}")
            st.stop()
            
        st.image(pil_image, caption='Uploaded X-ray', use_container_width=True)

        with st.spinner("Analyzing the image..."):
            # Resize to 224x224 (model input)
            processed_image = pil_image.resize((224, 224))
            img_array = np.array(processed_image, dtype=np.float32)

            # Add batch dimension and preprocess
            img_array = np.expand_dims(img_array, axis=0)
            img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

            # Debug: check shape
            st.write("Input shape to model:", img_array.shape)  # should be (1,224,224,3)

            # Prediction
            prediction = model.predict(img_array)[0][0]
            
            class_names = ['NORMAL', 'PNEUMONIA']
            predicted_class_index = int(prediction > 0.5)
            predicted_class = class_names[predicted_class_index]
            confidence = prediction if prediction > 0.5 else 1 - prediction

            st.subheader("Prediction Result")
            if predicted_class == 'PNEUMONIA':
                st.error(f"Prediction: **{predicted_class}** with {confidence:.2f} confidence")
            else:
                st.success(f"Prediction: **{predicted_class}** with {confidence:.2f} confidence")

            st.subheader("Model Explainability (Grad-CAM)")
            st.markdown("The heatmap below highlights the areas of the image most important for the model's decision.")

            try:
                heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=predicted_class_index)
                superimposed_image = display_gradcam(pil_image, heatmap)

                col1, col2 = st.columns(2)
                with col1:
                    st.image(pil_image, caption="Original Image", use_container_width=True)
                with col2:
                    st.image(superimposed_image, caption="Grad-CAM Heatmap Overlay", use_container_width=True)
            except Exception as e:
                st.error(f"Error generating Grad-CAM heatmap: {e}")

elif app_mode == "About":
    st.header("About This Project")
    st.markdown("""
    This application is a part of the "**A Deep Learning Approach for Chest X-Ray Pneumonia Detection**" project.
    
    **Technology Stack:**
    - **Language:** Python
    - **Libraries:** TensorFlow, Keras, NumPy, Streamlit, Matplotlib, PIL
    - **Model Architecture:** EfficientNetB0 (using transfer learning)
    - **Dataset:** Chest X-Ray Images (Pneumonia) from Kaggle
    
    This project demonstrates how deep learning, specifically a fine-tuned convolutional neural network, can be used for medical image analysis. The Grad-CAM visualization provides insights into how the model makes its predictions, enhancing trust and interpretability.
    """)
