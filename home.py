import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.title("Hello World!")

st.subheader("Welcome to Streamlit!")

st.markdown(
    """
    ### I love it!
    """
)
