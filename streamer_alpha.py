import cv2
import threading
import socket
import struct
import io
import json
import numpy
import pickle
import image_convert
import tensorflow as tf

graph = tf.Graph()

class Streamer_alpha (threading.Thread):
  def __init__(self, hostname, port):
    threading.Thread.__init__(self)

    self.hostname = hostname
    self.port = port
    self.connected = False
    self.jpeg = None

  def run(self):

    self.isRunning = True

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print('Socket created')

    s.bind((self.hostname, self.port))
    print('Socket bind complete')

    data = b""
    payload_size = struct.calcsize("L")

    s.listen(10)
    print('Socket now listening')

    while self.isRunning:

      conn, addr = s.accept()

      while True:

        while len(data) < payload_size:
          data += conn.recv(4096)
        # data = conn.recv(4096)

        if data:
          packed_msg_size = data[:payload_size]
          data = data[payload_size:]
          msg_size = struct.unpack("L", packed_msg_size)[0]

          while len(data) < msg_size:
            data += conn.recv(4096)

          frame_data = data[:msg_size]

          # memfile = io.BytesIO()
          # memfile.write(json.loads(frame_data).encode('latin-1'))
          # # memfile.write(json.loads(frame_data))
          # memfile.seek(0)
          # frame = numpy.load(memfile)

          data = data[msg_size:]
          frame = pickle.loads(frame_data, encoding='latin1')

          frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          frame = frame.reshape(1, 320, 480, 3)
          global graph
          with graph.as_default():
            frame = image_convert.model.predict(frame)
          frame = frame.reshape(320, 480)
          frame = frame * 255

          ret, jpeg = cv2.imencode('.jpg', frame)
          self.jpeg = jpeg

          self.connected = True

        else:
          conn.close()
          self.connected = False
          break

    self.connected = False

  def stop(self):
    self.isRunning = False

  def client_connected(self):
    return self.connected

  def get_jpeg(self):
    return self.jpeg.tobytes()
