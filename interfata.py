import tkinter as tk
from tkinter import ttk
import os
import tkinter.messagebox
from PIL import ImageTk, Image


root = tk.Tk()
root.title('Groot')

root.geometry('{}x{}'.format(800, 350))

ttk.Style().configure('green/black.TLabel', foreground='#178f27', background='black')
ttk.Style().configure('green/black.TButton', foreground='#178f27', background='black')
def opencreate():
    os.system('python3 create_gestures.pyw') 
def openload():
    os.system('python3 load_images.pyw') 
def opentrain():
    os.system('python3 cnn_model_train.pyw') 
def opentest():
    os.system('python3 final.pyw') 
def opensemne():
    os.system('python3 semne.py')  
def popupinfo():
    tk.messagebox.showinfo("Informatii",mesaj)

mesaj=""" Bună!

    Pentru început, este indicat să te uiți pe "Vizualizeaza semne" pentru a învața câteva din semnele ASL, iar după poți începe prin a crea semnele dorite prin apăsarea butonului de "Crează semne", programul se oprește singur după atingerea numărului maxim de poze setat.
    
    Apoi trebuie împărțite pozele în train, test și val pentru a putea algoritmul să le folosească, prin apăsarea butonului de "Împarte imaginile".
    
    După acest lucru, se poate realiza învățarea acestuia, prin apăsarea butonului "Antrenează algoritmul", în urma căruia va rezulta un grafic ce poate fi salvat în calculator sau doar vizualizat.
    
    Pentru a testa și pentru a verifica semnele, trebuie să apeși pe "Testează algoritmul". """

logo = ImageTk.PhotoImage(Image.open('/Users/teodorabotez/Desktop/Licenta/Cod/groot.jpeg'))

w1 = ttk.Label(root, image=logo).pack(side="right")
b1 = tk.Label(root,text="Bună!Mă numesc Groot și te voi ajuta să înveți limbajul semnelor!")
b1.place(x=430, y=30, anchor="center")
b2 = tk.Label(root,text="În primul rând trebuie să creezi gesturile, apăsând pe butonul de jos!")
b2.place(x=300, y=80, anchor="center")
buton = ttk.Button(root,text="Crează Semne",style='green/black.TButton', command=opencreate)
buton.place(x=300, y=110, anchor="center")
b3 = tk.Label(root,text="Apoi pozele trebuie să fie împărțite în test/train/val labels/images,apăsând butonul de jos!")
b3.place(x=300, y=140, anchor="center")
buton1 =ttk.Button(root,text="Împarte imaginile",style='green/black.TButton',command=openload)
buton1.place(x=300, y=170, anchor="center")
b4 = tk.Label(root,text="După trebuie învățat algoritmul,apăsând pe butonul de jos!")
b4.place(x=300, y=200, anchor="center")
buton2 = ttk.Button(root,text="Antrenează algoritmul",style='green/black.TButton',command=opentrain)
buton2.place(x=300, y=230, anchor="center")
b5 = tk.Label(root,text="Ultimul pas este testarea algoritmul și a pozelor realizate!")
b5.place(x=300, y=260, anchor="center")
buton3 = ttk.Button(root,text="Testează algoritmul",style='green/black.TButton',command=opentest)
buton3.place(x=300, y=290, anchor="center")
buton4 = ttk.Button(root,text="Vizualizeaza semnele",style='green/black.TButton',command=opensemne)
buton4.place(x=90, y=330, anchor="center")
buton5 = ttk.Button(root,text="Informatii",style='green/black.TButton',command=popupinfo)
buton5.place(x=750, y=330, anchor="center")

root.mainloop()























