import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load model
loaded_model = tf.keras.models.load_model('model/model_inceptionV3_epoch5.h5')

# Label mapping
label_mapping = {
    0: 'Bean',
    1: 'Bitter_Gourd',
    2: 'Bottle_Gourd',
    3: 'Brinjal',
    4: 'Broccoli',
    5: 'Cabbage',
    6: 'Capsicum',
    7: 'Carrot',
    8: 'Cauliflower',
    9: 'Cucumber',
    10: 'Papaya',
    11: 'Potato',
    12: 'Pumpkin',
    13: 'Radish',
    14: 'Tomato'
}

# Judul dan Logo GitHub dalam 1 baris horizontal
st.markdown("""
<div style="display: flex; align-items: center; justify-content: space-between;">
    <h1 style="margin-bottom: 0;">Klasifikasi Gambar Sayuran</h1>
    <a href="https://github.com/username/repo" target="_blank">
        <img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" style="height:40px; object-fit: contain;">
    </a>
</div>
""", unsafe_allow_html=True)

# Deskripsi singkat
st.markdown("""
Aplikasi ini melakukan klasifikasi gambar menjadi salah satu dari jenis sayuran berikut:

Bean, Bitter Gourd, Bottle Gourd, Brinjal, Broccoli, Cabbage, Capsicum, Carrot,  
Cauliflower, Cucumber, Papaya, Potato, Pumpkin, Radish, dan Tomato.

Unggah gambar sayuran, lalu tekan tombol **Prediksi** untuk melihat hasil klasifikasinya.
""")

# Upload gambar
uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png"])

# Tombol prediksi
if uploaded_file is not None:
    if st.button("Prediksi"):
        # Baca gambar
        image = Image.open(uploaded_file)
        
        # Preprocessing
        image_resized = image.resize((150, 150))
        image_array = np.array(image_resized) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # Prediksi
        predictions = loaded_model.predict(image_array)
        predicted_class = np.argmax(predictions)
        predicted_label = label_mapping[predicted_class]

        # Output
        st.image(image, caption="Gambar yang Diupload", use_container_width=True)
        st.markdown(f"### Hasil Prediksi: **{predicted_label}**")
