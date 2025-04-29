import streamlit as st

# Set up Streamlit app
st.set_page_config(
    layout="wide",
    page_title="AI Trading Assistant",
    page_icon="📊"
)

st.title("AI Trading Assistant")
st.write("Navigate between pages using the sidebar")

# Sidebar configuration
st.sidebar.success("Select a page above")