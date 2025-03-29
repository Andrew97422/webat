import streamlit as st
import requests
import io
from PIL import Image

st.set_page_config(page_title="Загрузка и получение изображений")
st.write("Загрузка и получение изображений")

url = 'http://localhost:8000/api/v1/images'

def send_upload(files):
    file_data = [('files', (file.name, file, file.type)) for file in files]
    response = requests.post(url + '/upload', files=file_data)
    return response.json()

def send_load():
    response = requests.post(url + '/load')
    if response.status_code == 200:
        # Возвращаем байтовые данные из response.content
        return response.content
    else:
        st.error("Failed to load image")
        return None

files = st.file_uploader("Choose your file...", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])

for file in files:
    if file is not None:
        st.image(file, caption=file.name)

if st.button("Send request for upload"):
    if files:
        response = send_upload(files)
        st.write(response)

if st.button("Send request for load"):
    image_bytes = send_load()
    if image_bytes:
        try:
            image = Image.open(io.BytesIO(image_bytes))
            st.image(image)
        except Exception as e:
            st.error(f"Error displaying image: {e}")