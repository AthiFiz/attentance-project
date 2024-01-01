import numpy as np
import pandas as pd
import cv2

import redis

from insightface.app import FaceAnalysis
from sklearn.metrics import pairwise

import time
from datetime import datetime

hostname = "redis-13392.c325.us-east-1-4.ec2.cloud.redislabs.com"
portnumber = 13392
password = "zzq1ZvmrXn80N8aeJwvaXl7PjNh6BTQf"

r = redis.StrictRedis(host=hostname,
                      port=portnumber,
                      password=password)

def retrive_data(name):
    retrive_dict = r.hgetall(name)
    retrive_series = pd.Series(retrive_dict)
    retrive_series = retrive_series.apply(lambda x: np.frombuffer(x, dtype=np.float32))
    index = retrive_series.index
    index = list(map(lambda x:x.decode(), index))
    retrive_series.index = index
    retrive_df = retrive_series.to_frame().reset_index()
    retrive_df.columns = (["name_role", "facial_features"])
    retrive_df[["name", "role"]] = retrive_df["name_role"].apply(lambda x: x.split("@")).apply(pd.Series)
    return retrive_df[["name", "role", "facial_features"]]



face_app = FaceAnalysis(name="buffalo_sc",
                        root="insightface_model",
                        providers=(["CPUExecutionProvider"]))
                        
face_app.prepare(ctx_id=0,
                 det_size=(640,640),
                 det_thresh=0.5)

def ml_search_algorithm(data_frame, feature_column, test_vector, thresh = 0.5, name_role=["name", "role"]):
    
    data_frame = data_frame.copy()
    X_list =data_frame[feature_column].tolist()
    x = np.asarray(X_list)

    similarity = pairwise.cosine_similarity(x, test_vector.reshape(1,-1))
    similar_arr = np.array(similarity).flatten()
    data_frame["cosine"] = similar_arr

    data_filter = data_frame.query(f"cosine >={thresh}")
    data_filter.reset_index(drop=True, inplace=True)
    if len(data_filter) > 0:
        argmax = data_filter["cosine"].argmax()
        person_name, person_role = data_filter.loc[argmax][name_role]
    else:
        person_name = "Unknown"
        person_role = "Unknown"

    return person_name, person_role


class RealTimePred:
    def __init__(self):
        self.logs = dict(name=[], role=[], current_time=[])

    def reset_dict(self):
        self.logs = dict(name=[], role=[], current_time=[])

    def save_logs(self):
        data_frame = pd.DataFrame(self.logs)

        data_frame.drop_duplicates("name", inplace=True)

        name_list = data_frame["name"].tolist()
        role_list = data_frame["role"].tolist()
        ctime_list = data_frame["current_time"].tolist()
        encoded_data = []

        for name, role, ctime in zip(name_list, role_list, ctime_list):
            if name != "Unknown":
                concat_string = f"{name}@{role}@{ctime}"
                encoded_data.append(concat_string)
        
        if len(encoded_data) > 0:
            r.lpush("attendance:logs", *encoded_data)

        self.reset_dict()

    def face_prediction(self ,test_img, 
                        data_frame, 
                        feature_column, 
                        thresh = 0.5, 
                        name_role=["name", "role"]):
        
        current_time = str(datetime.now())
        
        test_copy = test_img.copy()
        results = face_app.get(test_img)
        
        for res in results:
            x1, y1, x2, y2 = res["bbox"].astype(int)
            embeddings = res["embedding"]
        
            name, role = ml_search_algorithm(data_frame, 
                                            feature_column, 
                                            test_vector=embeddings,
                                            name_role=["name", "role"],
                                            thresh=thresh)
        
            if name == "Unknown":
                color = (0,0,255)
            else:
                color = (0,255,0)
        
            cv2.rectangle(test_copy, (x1,y1), (x2,y2), color, 1)
            cv2.putText(test_copy, name, (x1,y1-2), cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)
            cv2.putText(test_copy, current_time, (x1,y2+12), cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)

            self.logs["name"].append(name)
            self.logs["role"].append(role)
            self.logs["current_time"].append(current_time)

        return test_copy
    


class RegistrationForm:
    def __init__(self):
        self.sample = 0
    
    def reset(self):
        self.sample = 0

    def get_embeddings(self, frame):

        results = face_app.get(frame, max_num=1)
        embeddings = None

        for res in results:
            self.sample += 1
            x1, y1, x2, y2 = res["bbox"].astype(int)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 1)
            text = f"samples: {self.sample}"
            cv2.putText(frame, text, (x1,y1), cv2.FONT_HERSHEY_DUPLEX, 0.4, (255,255,0), 1)


            embeddings = res["embedding"]

        return frame, embeddings

                