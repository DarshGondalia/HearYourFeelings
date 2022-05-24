from PIL import Image
import cv2 as cv
import numpy
import os
import pywhatkit
from tensorflow import keras

def casc():
	face_cascade = cv.CascadeClassifier('/Users/manan/Desktop/389Final/haarcascade_frontalface_default.xml')
	cap = cv.VideoCapture(0)
	while 1:
		ret, img = cap.read()
		gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray, 1.3, 5)
		for (x, y, w, h) in faces:
			cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
			im = [x, y, w, h]
		cv.imshow('img', img)

		if cv.waitKey(30) & 0xFF == 27:
			break
		elif cv.waitKey(30) == 13:
			x, y, w, h = im
			cropped = img[y:y + h, x:x + w]
			cv.imwrite('/Users/manan/Desktop/389Final/assets/input_haar_cropped.jpeg', cropped)
			break
	
	cap.release()
	cv.destroyAllWindows()

def preprocess(img):
	image = Image.open(img)
	image = image.resize((48, 48))
	image = image.convert('L')
	image.save('/Users/manan/Desktop/389Final/assets/face_grayscale.jpeg')
	return numpy.array(image)

def youplay(emotion):
	emotions = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
	if emotion == "angry":
		pywhatkit.playonyt("https://www.youtube.com/watch?v=NvPZ-wvUDdM")
	elif emotion == "neutral":
		pywhatkit.playonyt("https://www.youtube.com/watch?v=Q6MemVxEquE&t=8385s")
	elif emotion == "fear":
		pywhatkit.playonyt("https://www.youtube.com/watch?v=MvHVzOREGUQ")
	elif emotion == "happy":
		pywhatkit.playonyt("https://www.youtube.com/watch?v=xiUhNx24iZs")
	elif emotion == "sad":
		pywhatkit.playonyt("https://www.youtube.com/watch?v=NkstXAUSpyM")
	elif emotion == "surprise":
		pywhatkit.playonyt("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
	elif emotion == "disgust":
		pywhatkit.playonyt("https://www.youtube.com/watch?v=VDa5iGiPgGs")
	return 0

def emotionprediction(face):
	emotions = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
	arr = cv.resize(face, (48, 48))
	arr = arr.astype("float") / 255.0
	arr = keras.preprocessing.image.img_to_array(arr)
	arr = numpy.expand_dims(arr, axis=0)	
	model = keras.models.load_model('/Users/manan/Desktop/389Final/my_model.h5')
			# print(model.summary())
	prediction = model.predict(arr)[0]
	maxindex = int(numpy.argmax(prediction))
	return emotions[maxindex]

def main():
	casc()
	img = '/Users/manan/Desktop/389Final/assets/input_haar_cropped.jpeg'
	face = preprocess(img)
	os.remove(img)

	emotion = emotionprediction(face)
	youplay(emotion)
	print("pe")
	
main()