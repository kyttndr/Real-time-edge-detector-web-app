import cv2
import numpy as np
import socket
import sys
import pickle
import struct
import io
import json
import pickle
import image_convert
import tensorflow as tf

cap = cv2.VideoCapture(0)
clientsocket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
clientsocket.connect(('localhost', 8089))

while(cap.isOpened()):
  ret, frame = cap.read()

  frame = cv2.resize(frame, (480, 320))

  # Canny
  # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  # print(frame.shape)
  # frame = cv2.Canny(frame, 100, 200)


  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  frame = frame.reshape(1, 320, 480, 3)
  # global graph
  with tf.get_default_graph().as_default():
  	frame = image_convert.model.predict(frame)
  frame = frame.reshape(320, 480)
  frame = frame * 255

  # memfile = io.BytesIO()
  # np.save(memfile, frame)
  # memfile.seek(0)
  # data = json.dumps(memfile.read().decode('latin-1'))
  data = pickle.dumps(frame)

  clientsocket.sendall(struct.pack("L", len(data)) + data)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
