from  Home import st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer
import av
from Home import face_rec

st.set_page_config(page_title="Registration Form", layout="centered")
st.subheader("Registration Form")

registration_form = face_rec.RegistrationForm()

person_name = st.text_input(label="Name", placeholder="First and Last Name")
role = st.selectbox(label="Select your Role", options=("Student", "Teacher"))

def video_callback(frame):

    img = frame.to_ndarray(format="bgr24") #3d bgr
    reg_img, embedding = registration_form.get_embeddings(img)

    if embedding is not None:
        with open("face_embedding.txt", mode="ab") as f:
            np.savetxt(f, embedding)

    return av.VideoFrame.from_ndarray(reg_img, format="bgr24")

webrtc_streamer(key="registration", video_frame_callback=video_callback)


if st.button("Submit"):
    st.write(f"Person Name = ", person_name )
    st.write(f"Your Role = ", role)