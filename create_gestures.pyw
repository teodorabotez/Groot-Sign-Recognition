import cv2
import numpy as np
import pickle, os, sqlite3, random

image_x, image_y = 50, 50

def get_hand_hist():
	with open("hist", "rb") as f:
		hist = pickle.load(f)
	return hist

def init_create_folder_database():
	if not os.path.exists("simbol"):
		os.mkdir("simbol")
	if not os.path.exists("simbol.db"):
		conn = sqlite3.connect("simbol.db")
		create_table_cmd = "CREATE TABLE simbol ( cod INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE, nume TEXT NOT NULL )"
		conn.execute(create_table_cmd)
		conn.commit()

def create_folder(folder_name):
	if not os.path.exists(folder_name):
		os.mkdir(folder_name)

def store_in_db(cod, nume):
	conn = sqlite3.connect("simbol.db")
	cmd = "INSERT INTO simbol (cod, nume) VALUES (%s, \'%s\')" % (cod, nume)
	try:
		conn.execute(cmd)
	except sqlite3.IntegrityError:
		choice = input("Codul introdus deja exista.Doriti sa il suprascrieti?(da/nu) ")
		if choice.lower() == 'da':
			cmd = "UPDATE simbol SET nume = \'%s\' WHERE cod = %s" % (nume, cod)
			conn.execute(cmd)
		else:
			print("Nu se intampla nimic!!!!")
			quit()
			return
	conn.commit()

def store_images(cod):
	total_pics = 1200
	hist = get_hand_hist()
	cam = cv2.VideoCapture(0)
	if cam.read()[0]==False:
		cam = cv2.VideoCapture(0)
	x, y, w, h = 300, 100, 300, 300 

	create_folder("simbol/"+str(cod))
	pic_no = 0
	flag_start_capturing = False
	frames = 0
	
	while True:
		img = cam.read()[1]
		img = cv2.flip(img,1)
		imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		dst = cv2.calcBackProject([imgHSV], [0, 1], hist, [0, 180, 0, 256], 1)
		disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
		cv2.filter2D(dst,-1,disc,dst)
		
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		gray = cv2.GaussianBlur(gray, (5, 5), 0)
		
		thresh = cv2.threshold(gray,45,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
		thresh = cv2.merge((thresh,thresh,thresh))
		thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
		img = cv2.resize(img, (600, 500))
		thresh = cv2.resize(thresh, (600, 500))
		contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]

		if len(contours) > 0:
			contour = max(contours, key = cv2.contourArea)

			if cv2.contourArea(contour) > 10000 and frames > 50:
				x1, y1, w1, h1 = cv2.boundingRect(contour)
				pic_no += 1
				save_img = thresh[y1:y1+h1, x1:x1+w1]
				if w1 > h1:
					save_img = cv2.copyMakeBorder(save_img, int((w1-h1)/2) , int((w1-h1)/2) , 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
				elif h1 > w1:
					save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1-w1)/2) , int((h1-w1)/2) , cv2.BORDER_CONSTANT, (255, 255, 255))
			
				save_img = cv2.resize(save_img, (image_x, image_y))
				cv2.putText(img, "Se realizeaza capturile pentru poze.", (30, 60), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255))
				cv2.imwrite("simbol/"+str(cod)+"/"+str(pic_no)+".jpg", save_img)
				
		thresh = cv2.resize(thresh,(0,0), None, 1.2, 1.2)
		thresh = cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR)
		img = cv2.resize(img,(0,0), None, 1.2, 1.2)

		#cv2.rectangle(img, (420,304), (700,100), (400,255,0), 2)
		cv2.putText(img, str(pic_no), (30, 400), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255))
		numpy_vertical_concat = np.concatenate((thresh ,img), axis = 1)
		
		numpy_vertical_concat = cv2.resize(numpy_vertical_concat,(0,0), None, 1, 1)
		v2 = np.hstack((thresh,img))
		cv2.imshow("Creare Gesturi",v2)

		
		keypress = cv2.waitKey(1)
		if keypress == ord('c'):
			if flag_start_capturing == False:
				flag_start_capturing = True
			else:
				flag_start_capturing = False
				frames = 0
				print("Numarul de poze pana in momentul opririi este de ",pic_no)
		if flag_start_capturing == True:
			frames += 1
		if pic_no == total_pics:
			print("Numarul total de poze a fost atins!S-au realizat",pic_no,"poze!")
		
			quit()

init_create_folder_database()
cod = input("Introduceti numarul gestului:  ")
nume = input("Introduceti numele gestului sau textul acestuia:  ")
store_in_db(cod, nume)
store_images(cod)
