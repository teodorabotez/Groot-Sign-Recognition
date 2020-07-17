import cv2, pickle
import numpy as np
import tensorflow as tf
import os
import sys
import sqlite3, pyttsx3
from keras.models import load_model
from threading import Thread
import scipy
import pygame
from time import sleep
import threading

pygame.init()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


model = load_model('/Users/teodorabotez/Desktop/Licenta/Cod/cnn_model_2020.h5')
def play_A():
	pygame.mixer.music.load("sunete/A.mp3")
	pygame.mixer.music.play()
	sleep(2)
	pygame.mixer.music.stop()
def play_B():
	pygame.mixer.music.load("sunete/B.mp3")
	pygame.mixer.music.play()
	sleep(2)
	pygame.mixer.music.stop()
def play_C():
	pygame.mixer.music.load("sunete/C.mp3")
	pygame.mixer.music.play()
	sleep(2)
	pygame.mixer.music.stop()
def play_D():
	pygame.mixer.music.load("sunete/D.mp3")
	pygame.mixer.music.play()
	sleep(2)
	pygame.mixer.music.stop()
def play_E():
	pygame.mixer.music.load("sunete/E.mp3")
	pygame.mixer.music.play()
	sleep(2)
	pygame.mixer.music.stop()
def play_F():
	pygame.mixer.music.load("sunete/F.mp3")
	pygame.mixer.music.play()
	sleep(2)
	pygame.mixer.music.stop()
def play_G():
	pygame.mixer.music.load("sunete/G.mp3")
	pygame.mixer.music.play()
	sleep(2)
	pygame.mixer.music.stop()
def play_H():
	pygame.mixer.music.load("sunete/H.mp3")
	pygame.mixer.music.play()
	sleep(2)
	pygame.mixer.music.stop()
def play_I():
	pygame.mixer.music.load("sunete/I.mp3")
	pygame.mixer.music.play()
	sleep(2)
	pygame.mixer.music.stop()
def play_J():
	pygame.mixer.music.load("sunete/J.mp3")
	pygame.mixer.music.play()
	sleep(2)
	pygame.mixer.music.stop()
def play_K():
	pygame.mixer.music.load("sunete/K.mp3")
	pygame.mixer.music.play()
	sleep(2)
	pygame.mixer.music.stop()
def play_L():
	pygame.mixer.music.load("sunete/L.mp3")
	pygame.mixer.music.play()
	sleep(2)
	pygame.mixer.music.stop()
def play_M():
	pygame.mixer.music.load("sunete/M.mp3")
	pygame.mixer.music.play()
	sleep(2)
	pygame.mixer.music.stop()
def play_N():
	pygame.mixer.music.load("sunete/N.mp3")
	pygame.mixer.music.play()
	sleep(2)
	pygame.mixer.music.stop()
def play_O():
	pygame.mixer.music.load("sunete/O.mp3")
	pygame.mixer.music.play()
	sleep(2)
	pygame.mixer.music.stop()
def play_P():
	pygame.mixer.music.load("sunete/P.mp3")
	pygame.mixer.music.play()
	sleep(2)
	pygame.mixer.music.stop()
def play_Q():
	pygame.mixer.music.load("sunete/Q.mp3")
	pygame.mixer.music.play()
	sleep(2)
	pygame.mixer.music.stop()
def play_R():
	pygame.mixer.music.load("sunete/R.mp3")
	pygame.mixer.music.play()
	sleep(2)
	pygame.mixer.music.stop()
def play_S():
	pygame.mixer.music.load("sunete/S.mp3")
	pygame.mixer.music.play()
	sleep(2)
	pygame.mixer.music.stop()
def play_T():
	pygame.mixer.music.load("sunete/T.mp3")
	pygame.mixer.music.play()
	sleep(2)
	pygame.mixer.music.stop()
def play_U():
	pygame.mixer.music.load("sunete/U.mp3")
	pygame.mixer.music.play()
	sleep(2)
	pygame.mixer.music.stop()
def play_V():
	pygame.mixer.music.load("sunete/V.mp3")
	pygame.mixer.music.play()
	sleep(2)
	pygame.mixer.music.stop()
def play_W():
	pygame.mixer.music.load("sunete/W.mp3")
	pygame.mixer.music.play()
	sleep(2)
	pygame.mixer.music.stop()
def play_X():
	pygame.mixer.music.load("sunete/X.mp3")
	pygame.mixer.music.play()
	sleep(2)
	pygame.mixer.music.stop()
def play_Y():
	pygame.mixer.music.load("sunete/Y.mp3")
	pygame.mixer.music.play()
	sleep(2)
	pygame.mixer.music.stop()
def play_Z():
	pygame.mixer.music.load("sunete/Z.mp3")
	pygame.mixer.music.play()
	sleep(2)
	pygame.mixer.music.stop()
def play_0():
	pygame.mixer.music.load("sunete/0.mp3")
	pygame.mixer.music.play()
	sleep(2)
	pygame.mixer.music.stop()
def play_1():
	pygame.mixer.music.load("sunete/1.mp3")
	pygame.mixer.music.play()
	sleep(2)
	pygame.mixer.music.stop()
def play_2():
	pygame.mixer.music.load("sunete/2.mp3")
	pygame.mixer.music.play()
	sleep(2)
	pygame.mixer.music.stop()
def play_3():
	pygame.mixer.music.load("sunete/3.mp3")
	pygame.mixer.music.play()
	sleep(2)
	pygame.mixer.music.stop()
def play_4():
	pygame.mixer.music.load("sunete/4.mp3")
	pygame.mixer.music.play()
	sleep(2)
	pygame.mixer.music.stop()
def play_5():
	pygame.mixer.music.load("sunete/5.mp3")
	pygame.mixer.music.play()
	sleep(2)
	pygame.mixer.music.stop()
def play_6():
	pygame.mixer.music.load("sunete/6.mp3")
	pygame.mixer.music.play()
	sleep(2)
	pygame.mixer.music.stop()
def play_7():
	pygame.mixer.music.load("sunete/7.mp3")
	pygame.mixer.music.play()
	sleep(2)
	pygame.mixer.music.stop()
def play_8():
	pygame.mixer.music.load("sunete/8.mp3")
	pygame.mixer.music.play()
	sleep(2)
	pygame.mixer.music.stop()
def play_9():
	pygame.mixer.music.load("sunete/9.mp3")
	pygame.mixer.music.play()
	sleep(2)
	pygame.mixer.music.stop()
def play_Casa():
	pygame.mixer.music.load("sunete/Casa.mp3")
	pygame.mixer.music.play()
	sleep(2)
	pygame.mixer.music.stop()
def play_Dragoste():
	pygame.mixer.music.load("sunete/Dragoste.mp3")
	pygame.mixer.music.play()
	sleep(2)
	pygame.mixer.music.stop()
def play_Mai_Mult():
	pygame.mixer.music.load("sunete/Mai-mult.mp3")
	pygame.mixer.music.play()
	sleep(2)
	pygame.mixer.music.stop()
def play_Prieten():
	pygame.mixer.music.load("sunete/Prieten.mp3")
	pygame.mixer.music.play()
	sleep(2)
	pygame.mixer.music.stop()
def play_Telefon():
	pygame.mixer.music.load("sunete/Telefon.mp3")
	pygame.mixer.music.play()
	sleep(2)
	pygame.mixer.music.stop()
def play_Al_lui_Al_ei():
	pygame.mixer.music.load("sunete/Al-lui-Al-ei.mp3")
	pygame.mixer.music.play()
	sleep(2)
	pygame.mixer.music.stop()
def play_Al_meu():
	pygame.mixer.music.load("sunete/Al-meu.mp3")
	pygame.mixer.music.play()
	sleep(2)
	pygame.mixer.music.stop()
def play_El_ea():
	pygame.mixer.music.load("sunete/El-ea.mp3")
	pygame.mixer.music.play()
	sleep(2)
	pygame.mixer.music.stop()
def play_EU():
	pygame.mixer.music.load("sunete/Eu.mp3")
	pygame.mixer.music.play()
	sleep(2)
	pygame.mixer.music.stop()

def paused():
	pygame.mixer.music.pause()

def get_hand_hist():
	with open("hist", "rb") as f:
		hist = pickle.load(f)
	return hist

def get_image_size():
	img = cv2.imread('simbol/1/1.jpg', 0)
	return img.shape

image_x, image_y = get_image_size()

def keras_process_image(img):
	img = cv2.resize(img, (image_x, image_y))
	img = np.array(img, dtype=np.float32)
	img = np.reshape(img, (1, image_x, image_y, 1))
	return img

def keras_predict(model, image):
	processed = keras_process_image(image)
	pred_probab = model.predict(processed)[0]
	pred_class = list(pred_probab).index(max(pred_probab))
	#print("pred_probab" + str(max(pred_probab)*100))
	return max(pred_probab), pred_class

def get_pred_text_from_db(pred_class):
	conn = sqlite3.connect("simbol.db")
	cmd = "SELECT nume FROM simbol WHERE cod="+str(pred_class)
	cursor = conn.execute(cmd)
	for row in cursor:
		return row[0]

def get_pred_from_contour(contour, thresh):
	x1, y1, w1, h1 = cv2.boundingRect(contour)
	save_img = thresh[y1:y1+h1, x1:x1+w1]
	text = ""
	if w1 > h1:
		save_img = cv2.copyMakeBorder(save_img, int((w1-h1)/2) , int((w1-h1)/2) , 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
	elif h1 > w1:
		save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1-w1)/2) , int((h1-w1)/2) , cv2.BORDER_CONSTANT, (0, 0, 0))
	pred_probab, pred_class = keras_predict(model, save_img)
	pred_text = pred_probab*100
	if pred_probab*100 > 70:
		text = get_pred_text_from_db(pred_class)
	return text,pred_text

hist = get_hand_hist()
x, y, w, h = 300, 100, 300, 300

def get_img_contour_thresh(img):
	img = cv2.flip(img, 1)
	imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	dst = cv2.calcBackProject([imgHSV], [0, 1], hist, [0, 180, 0, 256], 1)
	disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
	cv2.filter2D(dst,-1,disc,dst)

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	thresh = cv2.threshold(gray,45,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
	thresh = cv2.merge((thresh,thresh,thresh))
	thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
	
	contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
	return img, contours, thresh

def paused():
    
    pygame.mixer.music.pause()


def text_mode(cam):
	text = ""
	word = ""
	perce = ""
	q = 0
	count_same_frame = 0
	while True:
		img = cam.read()[1]
		img = cv2.resize(img, (600, 480))
		img, contours, thresh = get_img_contour_thresh(img)
		old_text = text
		if len(contours) > 0:
			contour = max(contours, key = cv2.contourArea)
			if cv2.contourArea(contour) > 10000:
				ptext = get_pred_from_contour(contour, thresh)
				text = ptext[0]
				q = ptext[1]
				if text is not None:
					if old_text == text:
						count_same_frame += 1
					else:
						count_same_frame = 0

					if count_same_frame > 25:
						word = word + text
						pygame.init()

						if text == "A":
							tA = threading.Thread(target=play_A)
							tA.start()	
						elif text == "B":
							tB = threading.Thread(target=play_B)
							tB.start()
						elif text == "C":
							tC = threading.Thread(target=play_C)
							tC.start()
						elif text == "D":
							tD = threading.Thread(target=play_D)
							tD.start()
						elif text == "E":
							tE = threading.Thread(target=play_E)
							tE.start()
						elif text == "F":
							tF = threading.Thread(target=play_F)
							tF.start()
						elif text == "G":
							tG = threading.Thread(target=play_G)
							tG.start()
						elif text == "H":
							tH = threading.Thread(target=play_H)
							tH.start()
						elif text == "I":
							tI = threading.Thread(target=play_I)
							tI.start()
						elif text == "J":
							tJ = threading.Thread(target=play_J)
							tJ.start()
						elif text == "K":
							tK = threading.Thread(target=play_K)
							tK.start()
						elif text == "L":
							tL = threading.Thread(target=play_L)
							tL.start()
						elif text == "M":
							tM = threading.Thread(target=play_M)
							tM.start()
						elif text == "N":
							tN = threading.Thread(target=play_N)
							tN.start()
						elif text == "O":
							tO = threading.Thread(target=play_O)
							tO.start()
						elif text == "P":
							tP = threading.Thread(target=play_P)
							tP.start()
						elif text == "Q":
							tQ = threading.Thread(target=play_Q)
							tQ.start()
						elif text == "R":
							tR = threading.Thread(target=play_R)
							tR.start()
						elif text == "S":
							tS = threading.Thread(target=play_S)
							tS.start()
						elif text == "T":
							tT = threading.Thread(target=play_T)
							tT.start()
						elif text == "U":
							tU = threading.Thread(target=play_U)
							tU.start()
						elif text == "V":
							tV = threading.Thread(target=play_V)
							tV.start()
						elif text == "W":
							tW = threading.Thread(target=play_W)
							tW.start()
						elif text == "X":
							tX = threading.Thread(target=play_X)
							tX.start()
						elif text == "Y":
							tY = threading.Thread(target=play_Y)
							tY.start()
						elif text == "Z":
							tZ = threading.Thread(target=play_Z)
							tZ.start()
						elif text == "0":
							t0 = threading.Thread(target=play_0)
							t0.start()
						elif text == "1":
							t1 = threading.Thread(target=play_1)
							t1.start()
						elif text == "2":
							t2 = threading.Thread(target=play_2)
							t2.start()
						elif text == "3":
							t3 = threading.Thread(target=play_3)
							t3.start()
						elif text == "4":
							t4 = threading.Thread(target=play_4)
							t4.start()
						elif text == "5":
							t5 = threading.Thread(target=play_5)
							t5.start()
						elif text == "6":
							t6 = threading.Thread(target=play_6)
							t6.start()
						elif text == "8":
							t8 = threading.Thread(target=play_8)
							t8.start()
						elif text == "9":
							t9 = threading.Thread(target=play_9)
							t9.start()
						elif text == "Casa":
							tCasa = threading.Thread(target=play_Casa)
							tCasa.start()
						elif text == "Dragoste":
							tDragoste = threading.Thread(target=play_Dragoste)
							tDragoste.start()
						elif text == "Mai mult":
							tMaimult = threading.Thread(target=play_Mai_Mult)
							tMaimult.start()
						elif text == "Telefon":
							tTelefon = threading.Thread(target=play_Telefon)
							tTelefon.start()
						elif text == "Prieten":
							tPrieten = threading.Thread(target=play_Prieten)
							tPrieten.start()
						elif text == "Al meu":
							tAM = threading.Thread(target=play_Al_meu)
							tAM.start()
						elif text == "Al lui/Al ei":
							tLE = threading.Thread(target=play_Al_lui_Al_ei)
							tLE.start()
						elif text == "Eu":
							tEu = threading.Thread(target=play_EU)
							tEu.start()
						elif text == "EL/Ea":
							tELEA = threading.Thread(target=play_El_ea)
							tELEA.start()		
						cv2.putText(blackboard, "Textul este " + word, (30, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 0))
						count_same_frame = 0
						
				elif count_same_frame < 20:
					text = ""
					word = ""
			elif cv2.contourArea(contour) < 1000:
				text = ""
				word = ""
		else:
			text = ""
			word = ""

		img = cv2.resize(img, (900, 550))
		thresh = cv2.resize(thresh, (900, 550))
		blackboard = np.zeros((480, 600, 3), dtype=np.uint8)
		blackboard = cv2.resize(blackboard, (2160,720))
		cv2.putText(blackboard, "Prediction= "+ (str(q)[:5])+ "%", (50, 60), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255,255))
		cv2.putText(blackboard, "Textul este " + text , (50, 130), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255))
		cv2.putText(blackboard, word, (50, 200), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255))
		#cv2.rectangle(img, (300,150), (550,300), (0,255,0), 2)
		thresh = cv2.resize(thresh,(0,0), None, 1.2, 1.2)
		thresh = cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR)
		img = cv2.resize(img,(0,0), None, 1.2, 1.2)
		blackboard = cv2.resize(blackboard,(0,0),None , 1 , 1)
		numpy_vertical_concat = np.concatenate((thresh ,img), axis = 1)
		numpy_vertical_concat = cv2.resize(numpy_vertical_concat,(0,0), None, 1, 1)
		v2 = np.hstack((img,thresh))
		v2 = cv2.resize(v2,(0,0), None, 1, 1)
		numpy_horizontal_concat = np.vstack((v2 ,blackboard))
		numpy_horizontal_concat = cv2.resize(numpy_horizontal_concat,(0,0), None, .5, .5)
		cv2.imshow("Recunoastere gesturi", numpy_horizontal_concat)
		keypress = cv2.waitKey(1)

	if keypress == ord('c'):
		return 2
	else:
		return 0

def recognize():
	cam = cv2.VideoCapture(0)
	
	if cam.read()[0]==False:
		cam = cv2.VideoCapture(0)
	text = ""
	word = ""
	count_same_frame = 0
	keypress = 1
	while True:
		if keypress == 1:
			keypress = text_mode(cam)
			break		

keras_predict(model, np.zeros((50, 50), dtype = np.uint8))		
recognize()