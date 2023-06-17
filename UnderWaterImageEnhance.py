from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import cv2
import os
import ntpath
import numpy as np
from PIL import Image
from os.path import join, exists
from keras.models import model_from_json
from utils.data_utils import getPaths, read_and_resize, preprocess, deprocess

main = tkinter.Tk()
main.title("Underwater Image Enhancement for Improved Visual Perception")
main.geometry("1300x1200")

global filename
global model

def loadModel():
    text.delete('1.0', END)
    global model
    with open('models/gen_p/model_15320_.json', "r") as json_file:
        loaded_model_json = json_file.read()
    json_file.close()    
    model = model_from_json(loaded_model_json)
    model.load_weights('models/gen_p/model_15320_.h5')
    text.insert(END,"FUnIE-GAN Model loaded\n")
    pathlabel.config(text="FUnIE-GAN Model loaded")

def uploadImage():
    global filename
    filename = filedialog.askopenfilename(initialdir="testImages")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n\n");
    

def generateImprovedImage():
    global filename
    global model
    inp_img = read_and_resize(filename, (256, 256))
    im = preprocess(inp_img)
    im = np.expand_dims(im, axis=0) 
    gen = model.predict(im)
    gen_img = deprocess(gen)[0]
    inputImg = cv2.imread(filename)
    inputImg = cv2.resize(inputImg,(256,256))
    cv2.imshow("Original Poor Quality Image",inputImg)
    cv2.moveWindow("Original Poor Quality Image", 450, 450)
    gen_img = cv2.cvtColor(gen_img, cv2.COLOR_RGB2BGR)
    gen_img = cv2.imshow("Enhance Generated Image",gen_img)
    gen_img = cv2.moveWindow("Enhance Generated Image", 750, 450)
    output_folder = 'output_enhance/'
    if not exists(output_folder):
        os.makedirs(output_folder)
    output_path = join(output_folder, ntpath.basename(filename))
    cv2.imwrite(output_path, gen_img)
    text.insert(END,"Enhanced Image saved in "+output_path+"\n")
    cv2.waitKey(0)
    
def close():
    main.destroy()

    
    
font = ('times', 16, 'bold')
title = Label(main, text='Underwater Image Enhancement for Improved Visual Perception',anchor=W)
title.config(bg='white', fg='black')  

title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)


font1 = ('times', 14, 'bold')
upload = Button(main, text="Generate & Load FUnIE-GAN Model", command=loadModel)
upload.place(x=50,y=100)
upload.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='chocolate', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=50,y=150)

markovButton = Button(main, text="Upload Poor Quality Image", command=uploadImage)
markovButton.place(x=50,y=200)
markovButton.config(font=font1)

lexButton = Button(main, text="Improved Visual Perception", command=generateImprovedImage)
lexButton.place(x=50,y=250)
lexButton.config(font=font1)

predictButton = Button(main, text="Exit", command=close)
predictButton.place(x=50,y=300)
predictButton.config(font=font1)


font1 = ('times', 12, 'bold')
text=Text(main,height=15,width=78)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=450,y=100)
text.config(font=font1)


main.config(bg='chocolate')
main.mainloop()
