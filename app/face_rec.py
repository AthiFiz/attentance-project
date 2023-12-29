import numpy as np
import pandas as pd
import cv2

import redis

from insightface.app import FaceAnalysis
from sklearn.metrics import pairwise

hostname = "redis-13392.c325.us-east-1-4.ec2.cloud.redislabs.com"
portnumber = 13392
password = "zzq1ZvmrXn80N8aeJwvaXl7PjNh6BTQf"

r = redis.StrictRedis(host=hostname,
                      port=portnumber,
                      password=password)

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


def face_prediction(test_img, data_frame, 
                    feature_column, 
                    thresh = 0.5, 
                    name_role=["name", "role"]):
    
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

    return test_copy
                