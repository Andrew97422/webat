import streamlit as st
import requests

st.set_page_config(page_title="Загрузка и получение документов")
st.write('### Something...')

url = 'http://localhost:8000/api/v1'

def send_upload(files):
    file_data = [('files', (file.name, file, file.type)) for file in files]
    response = requests.post(url + '/upload', files=file_data)
    return response.json()

def send_load():
    response = requests.post(url + '/load')
    return response.json()

files = st.file_uploader("Choose your file...", accept_multiple_files=True)

if st.button("Send request for upload"):
    if files:
        response = send_upload(files)
        st.write(response)

if st.button("Send request for load"):
    response = send_load()
    st.write(response)