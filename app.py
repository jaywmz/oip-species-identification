import streamlit as st

st.title("Species ID POC")
upload = st.file_uploader("📸 Upload a photo", type=["jpg","png"])
if upload:
    st.image(upload, caption="Your photo")
    st.write("…insert inference here…")
