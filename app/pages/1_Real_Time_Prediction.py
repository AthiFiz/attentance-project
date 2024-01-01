from Home import st, face_rec
from streamlit_webrtc import webrtc_streamer
import av
import time

st.set_page_config(page_title="Predictions", layout="centered")
st.subheader("Real-Time Attendance System")

with st.spinner("Retrieving Data from Redis DB"):
    redis_face_db = face_rec.retrive_data(name="academy:register")
    st.dataframe(redis_face_db)
st.success("Successfully Data Retrieved")

wait_time = 30
set_time = time.time()
real_time_pred = face_rec.RealTimePred()

def video_frame_callback(frame):

    global set_time
    
    img = frame.to_ndarray(format="bgr24")
    pred_image = real_time_pred.face_prediction(img, redis_face_db, "facial_features")

    time_now = time.time()
    time_diff = time_now - set_time
    if time_diff >= wait_time:
        real_time_pred.save_logs()
        set_time = time.time()
        print("Save Data to Redis Data Base")

    return av.VideoFrame.from_ndarray(pred_image, format="bgr24")

webrtc_streamer(key="realtimeprediction", video_frame_callback=video_frame_callback)