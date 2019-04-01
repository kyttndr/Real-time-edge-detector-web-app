from flask import Flask, render_template, Response
from streamer import Streamer
from streamer_alpha import Streamer_alpha
from streamer_canny import Streamer_canny

app = Flask(__name__)

def gen():
	streamer = Streamer('0.0.0.0', 8089)
	streamer.start()

	while True:
		if streamer.client_connected():
			frame = streamer.get_jpeg()
			yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def gen_alpha():
	streamer = Streamer_alpha('0.0.0.0', 8087)
	streamer.start()

	while True:
		# print(streamer.get_jpeg())
		if streamer.client_connected():
			frame = streamer.get_jpeg()
			yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def gen_canny():
	streamer = Streamer_canny('0.0.0.0', 8088)
	streamer.start()

	while True:
		# print(streamer.get_jpeg())
		if streamer.client_connected():
			frame = streamer.get_jpeg()
			yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/video_feed')
def video_feed():
	return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_alpha')
def video_feed_alpha():
	return Response(gen_alpha(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_canny')
def video_feed_canny():
	return Response(gen_canny(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
	app.run(host='0.0.0.0', threaded=True)
