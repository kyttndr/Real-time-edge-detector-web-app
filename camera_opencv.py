import cv2
from base_camera import BaseCamera
import image_convert
import cv2
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

graph = tf.get_default_graph()

class Camera(BaseCamera):
    video_source = 0

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
        camera = cv2.VideoCapture(Camera.video_source)
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        while True:
            # read current frame
            _, img = camera.read()

            # reshape
            img = cv2.resize(img, (480, 320))

            # Canny
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # img = cv2.Canny(img, 100, 200)

            # Ours
            # start_time = time.time()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.reshape(1, 320, 480, 3)
            global graph
            with graph.as_default():
                img = image_convert.model.predict(img)
            img = img.reshape(320, 480)
            img = img * 255

            # encode as a jpeg image and return it
            yield cv2.imencode('.jpg', img)[1].tobytes()
