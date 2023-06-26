import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np
import mediapipe as mp

#-----------------------------------------------
import os
from twilio.rest import Client
account_sid = os.environ['TWILIO_ACCOUNT_SID']
auth_token = os.environ['TWILIO_AUTH_TOKEN']
client = Client(account_sid, auth_token)
token = client.tokens.create()
#-----------------------------------------------

pose = mp.solutions.pose.Pose()

font = cv2.FONT_HERSHEY_SIMPLEX

st.title("Raise Your Hand")

class VideoProcessor:  
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        #------------------------------------------
        img = cv2.flip(img,1)            
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)

        h, w, c = img.shape

        if results.pose_landmarks:
            poseLms = results.pose_landmarks

            point0 = poseLms.landmark[0]
            cx0, cy0 = int(point0.x*w), int(point0.y*h)
            cv2.circle(img,(cx0,cy0),15,(0,0,255),-1)

            point19 = poseLms.landmark[19]
            cx19, cy19 = int(point19.x*w), int(point19.y*h)
            cv2.circle(img,(cx19,cy19),15,(0,255,0),-1)

            point20 = poseLms.landmark[20]
            cx20, cy20 = int(point20.x*w), int(point20.y*h)
            cv2.circle(img,(cx20,cy20),15,(0,255,255),-1)

            if (cy20 < cy0) and (cy19 > cy0):
                cv2.putText(img,"Left",(50,100),font,2,(0,255,255),8)
            elif (cy19 < cy0) and (cy20 > cy0):
                cv2.putText(img,"Right",(50,100),font,2,(0,255,0),8)
            else:
                cv2.putText(img,"???",(50,100),font,2,(255,0,255),8)      
        #------------------------------------------
        return av.VideoFrame.from_ndarray(img,format="bgr24")

webrtc_streamer(key="test",
                video_processor_factory=VideoProcessor,
                media_stream_constraints={"video": True,"audio": False},
                rtc_configuration={"iceServers":token.ice_servers}) ###
