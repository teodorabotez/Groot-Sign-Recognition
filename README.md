# Groot-Sign-Recognition
This project is based on another project which i modified and improved for my senior thesis.
The original project is https://github.com/harshbg/Sign-Language-Interpreter-using-Deep-Learning and i can't take fully credit for it.

After you clone the project, you need to install the following packages:

python3,tkinter
,pil
,cv2
,numpy
,pickle
,glob
,random
,sklearn
,tensorflow
,keras
,sqlite3
,threading
,pygame
,matplotlib

After you install those with pip3 install name_of_the_package, you need to create a histogram with the file : set_hand_histogram.py.

You'll run the file interfata.py to make the interface of the app appear.Then you need to create gestures by pressing the button "creare gesturi" Once you run this file, it requires you to fill the id of the gesture and the name of it.

The script will stop after the number of pics is reached.

To create test, train and val labels and images, you need to press the button "imparte imagini".

After that you need to train it by pressing "invata algoritmul" and after is done, a graph will appear. You can save it or look at it as it is.

After all those steps, you need to press "testeaza semnele".

The 2 buttons "vzualizeaza semne" and "informatii" are there to help you use the app better.

The first one shows you the gestures learned for this app and the second one shows you a pop-up message that presents you infos about how to use the app.

In the "sunete"folder, are the sounds that i recorded to play when a sign is recognized.

And in the "poze-semne" are the gestures required in the "vizualizeaza semne" window.
