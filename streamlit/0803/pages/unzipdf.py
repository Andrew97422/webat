import streamlit as st
import os.path as path

import zipfile
import pandas as pd

def unzip():
    with zipfile.ZipFile(path.abspath('archive.zip'), 'r') as zip_ref:
        zip_ref.extractall()

if st.button('unzip'):
    unzip()
    st.dataframe(pd.read_csv('train_and_test2.csv'))