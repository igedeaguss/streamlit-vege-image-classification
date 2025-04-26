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

# Buat dua kolom: satu untuk judul, satu untuk logo
col1, col2 = st.columns([9, 1])  # rasio 9:1 supaya lebih lebar ke judul
# Judul aplikasi
with col1:
    st.title("Klasifikasi Gambar Sayuran")
# Logo GitHub + Link
with col2:
    st.markdown("""
    <div style="text-align: right;">
        <a href="https://github.com/igedeaguss/vegetable-image-classification/tree/main" target="_blank">
            <img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" width="30" height="30">
        </a>
    </div>
    """, unsafe_allow_html=True)

# Deskripsi singkat
st.markdown("""
Aplikasi ini melakukan klasifikasi gambar menjadi salah satu dari jenis sayuran berikut:

🟢Bean, Bitter Gourd, Bottle Gourd, Brinjal, Broccoli, Cabbage, Capsicum, Carrot,  
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
