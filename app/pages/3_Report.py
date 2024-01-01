from Home import st, face_rec

st.set_page_config(page_title="Report", layout="wide")
st.subheader("Report")


name = "attendance:logs"

def load_logs(name, end=-1):
    log_list = face_rec.r.lrange(name, start=0, end=end)
    return log_list

tab1, tab2 = st.tabs(["Registered Data", "Logs"])

with tab1:
    if st.button("Refresh Data"):
        with st.spinner("Retrieving Data from Redis DB"):
            redis_face_db = face_rec.retrive_data(name="academy:register")
            st.dataframe(redis_face_db[["name", "role"]])

with tab2:
    if st.button("Refresh Logs"):
        st.write(load_logs(name=name))

