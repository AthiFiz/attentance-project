import streamlit as st


st.set_page_config(page_title="Attendance System", layout="wide")

st.header("Attendance System using Face Recognition")

with st.spinner("Loadeng Models and Connecting to Redis Data Base...."):
    import face_rec

st.success("Model Loaded Successfully")
st.success("Redis Dattabase is sucessfuly connected")