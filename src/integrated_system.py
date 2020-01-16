# GUI
from tkinter import *
from tkinter import font, Canvas, Button, Label, CENTER
# image manipualtion
from PIL import Image, ImageTk
# openCV for video analysis
import cv2
# cairo for drawing the output image
from cairo import * 
# python packages
import sys
from time import sleep
import numpy as np 
import noise 
# tensorflow for CNN
import tensorflow as tf
from tensorflow.keras import backend as K 
from tensorflow.keras.models import model_from_json 

# create the main window
root = Tk()
root.attributes('-fullscreen',True)
root.bind("<Escape>", lambda event: root.destroy())
root.title('Emotions visualizer')

# create label to hold the background image
image_output = Label(root, bg="black")
image_output.pack(expand=True, fill="both")

# welcome window with START button and exit info
a100 = font.Font(family='Arial', size=100)
button_start=Button(root,text="START", bg="black",fg="white",highlightbackground="black",bd="5",font=a100,relief="flat",command=lambda: start_video(button_start))
button_start.place(relx=0.5, rely=0.5, anchor=CENTER)
a20 = font.Font(family='Arial', size=20)
label_exit=Label(root,text="[ESC] to exit",bg="black",fg="white",highlightbackground="black",bd="5",font=a20,)
label_exit.place(relx=0,rely=0,anchor="nw")

# create label to hold the webcam image
web_frame=Frame(root)
web_frame.place(anchor="se", relx=1, rely=1)
image_webcam = Label(web_frame)
image_webcam.pack()

# load neural network model from json 
json_file = open("C:/Users/Gosia/Documents/CS/Semester 5 KTH/Artificial Intelligence/Project/final version/model.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("C:/Users/Gosia/Documents/CS/Semester 5 KTH/Artificial Intelligence/Project/final version/model.h5")
print("Loaded model from disk")

# read the classifiers for detecting the face
faceCascade = cv2.CascadeClassifier("C:/Users/Gosia/Documents/CS/Semester 5 KTH/Artificial Intelligence/Project/final version/haarcascade_frontalface_default.xml")

# Capture from camera
video_capture = cv2.VideoCapture(0)

#global variables (GV) for managing the main loop
img_counter=0
timer=0
x=0;y=0;h=0;w=0
width  = video_capture.get(3) 
height = video_capture.get(4)
sx=(int)(root.winfo_screenwidth())
sy=(int)(root.winfo_screenheight())

# GV for starting with a neutral emotion
emotion_index= 6
emotion_prob=0.5

# creating cairo surface and context for drawing the output art and painting the background black
surface = ImageSurface(FORMAT_ARGB32, sx, sy)
context = Context(surface)
context.set_source_rgb(0,0,0)
context.rectangle(0,0,sx,sy)
context.fill()
context.set_source_rgb(1,1,1)

# function for clicking start and opening the webcam
def start_video(button):
    button.place_forget()
    label_exit.place_forget()
    video_stream()

# function for video streaming
def video_stream():
    global timer
    global x,y,w,h
    global width,height
    global emotions
    global emotion_index,emotion_prob
    global context,surface

    #if no webcam available
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)

    #capture face
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(48, 48)
        )
    
    #draw rectangle around the face
    for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
    
    #frame image to webcam display
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    cv2image=cv2.flip(cv2image,1)
    img = Image.fromarray(cv2image)
    img=img.resize((int(width/3),int(height/3)))
    imgtk = ImageTk.PhotoImage(image=img)
    image_webcam.imgtk = imgtk
    image_webcam.configure(image=imgtk)

    # determine an emotion every 5 seconds
    if timer>50 :
        timer=0
        crop_frame = frame[y:y + h, x:x + w]
        #if there is a face detected
        if len(faces)>0:
            #adjust image to network format
            crop_grey_frame = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2GRAY)
            crop_grey_resized_frame = cv2.resize(crop_grey_frame, (48, 48))
            crop_grey_resized_frame= crop_grey_resized_frame.reshape(1, 48, 48, 1)

            # proccess the image with the network model and get the result
            predictions = loaded_model.predict(crop_grey_resized_frame)
            emotion_old_index=emotion_index
            emotion_index = np.where(predictions[0] == np.amax(predictions[0]))
            emotion_index= int(emotion_index[0])
            emotion_prob=np.amax(predictions[0])

    timer+=1

    # art part
    draw(z_increment[emotion_index],r_factor[emotion_index],colors[emotion_index],emotion_prob,wg[emotion_index],speed[emotion_index])
    

    image_webcam.after(10, video_stream) 

# GV which are the interpretations of the output of the neural net: 
# color, z-axis of the Perlin noise, weight of the brush, speed of the changing Perlin noise, addidtional multiplication factor
emotions=["Angry","Disgusted","Scared","Happy","Sad","Surprised","Neutral"]
colors=[(172,54,76),(213,138,227),(249,248,113),(255,162,144),(112,144,235),(255,139,191),(233,239,255)]
z_increment=[0.07,0.1,0.07,0.001,0.2,0.01,0.01]
wg=[1.5,2,1.5,1.7,1.7,1.5,1.7]
speed=[6,4,6,4,4,6,5]
r_factor=[8,7,5,5,3,3,3]

# a class which represents each of the lines on the output image
class Line:

    def __init__(self,sizex,sizey):
        self.x=0
        self.y=0
        self.old_x=0
        self.old_y=0
        self.dirx=0
        self.diry=0
        self.velx=0
        self.velx=0
        self.x=np.random.uniform(sizex)
        self.y=np.random.uniform(sizey)
        self.old_x=self.x
        self.old_y=self.y
        self.speed=3
  
    # determine the displacement of the next part of the line
    def move(self,z,inc,r):
        angle=(noise.pnoise3(self.x*inc,self.y*inc,z)*2*np.pi)*r
        self.dirx=np.cos(angle)
        self.diry=np.sin(angle)
        self.velx=self.dirx*self.speed
        self.vely=self.diry*self.speed
        self.old_x=self.x
        self.old_y=self.y
        self.x+=self.velx
        self.y+=self.vely
  
    #draw the next move of the line
    def display(self,op,wg,col):
        global context
        context.set_line_width(wg)
        (r,g,b)=col
        context.set_source_rgba(r/255,g/255,b/255,0.4)
        context.move_to(self.old_x,self.old_y)
        context.line_to(self.x,self.y)
        context.stroke()

    #check if the line is out of the display
    def check_edge(self,sizex,sizey):
        if (int)(self.x)>(int)(sizex) | (int)(self.y)>(int)(sizey) | (int)(self.x)<0 | (int)(self.y)<0:
            self.x=np.random.uniform(sizex)
            self.y=np.random.uniform(sizey)
            self.old_x=self.x
            self.old_y=self.y

# GV as parameters for the output image
inc=0.01
zoff=0
count=500

# array with the Line objects
lines=[]
for i in range(count):
    lines.append(Line(sx,sy))

# image output drawing function
def draw(z_inc,r,c,o,w,s_inc):
    global sx,sy,inc,zoff,context,surface,image_output

    for l in lines:
        l.speed=s_inc
        l.move(zoff,inc,r)
        l.display(o,w,c)
        l.check_edge(sx,sy)

    # converting from cairo image into a screen output
    img= Image.frombuffer("RGBA", (sx, sy), bytes(surface.get_data()), "raw", "BGRA", 0, 1)
    imgtk=ImageTk.PhotoImage(image=img)
    image_output.imgtk = imgtk
    image_output.configure(image=imgtk)

    zoff+=z_inc


# keep the main window open
root.mainloop()