from tkinter import *
import cv2
from PIL import Image, ImageTk
from keras.models import load_model
import numpy as np
model = load_model('model.h5') # load CNN model
auto_mode = False # manual mode at the beginning
count = 0
predict_delay_time = 6 # 60ms
display_accuracy = 3 # 0.001
my_dict = {1:'apple', 2:'banana', 3:'grapes', 4:'orange', 5:'pear', 6:'tomato'}
cap = cv2.VideoCapture(0) # turn on camera

################################################################ Functions ################################################################
def close_window():
    root.destroy()

def switch_mode():
    global auto_mode
    # switch to auto
    if auto_mode==False:
        auto_mode = True
        mode_but.config(image=auto_icon)
        print('Mode switched: Automatic')
    # switch to manual
    else:
        auto_mode = False
        mode_but.config(image=manu_icon)
        print('Mode switched: Manual')

def take_photos_manualy():
    if auto_mode==False:
        _, frame = cap.read()
        print('photo is taked!')
        cv2.imwrite("current_img.jpg", frame)
        current_photo = Image.open('current_img.jpg')
        img_predict(current_photo)

def take_photos_automatically():
    if auto_mode==True:
        _, frame = cap.read()
        cv2.imwrite("current_img.jpg", frame)
        current_photo = Image.open("current_img.jpg")
        img_predict(current_photo)

def config_bar(index, value):
    if index==1:
        canvas.coords(bar1, 0, 10, int(value*max_len), 20)               # config apple bar
    elif index==2:
        canvas.coords(bar2, 0, 40+gap, int(value*max_len), 50+gap)       # config banana bar
    elif index==3:
        canvas.coords(bar3, 0, 70+2*gap, int(value*max_len), 80+2*gap)   # config grapes bar
    elif index==4:
        canvas.coords(bar4, 0, 100+3*gap, int(value*max_len), 110+3*gap) # config orange bar
    elif index==5:
        canvas.coords(bar5, 0, 130+4*gap, int(value*max_len), 140+4*gap) # config pear bar
    elif index==6:
        canvas.coords(bar6, 0, 160+5*gap, int(value*max_len), 170+5*gap) # config tomato bar

def config_label(index, new_text):
    if index==1:
        apple_value_label.config(text=new_text)    # config apple value
    elif index==2:
        banana_value_label.config(text=new_text)   # config banana value
    elif index==3:
        grapes_value_label.config(text=new_text)   # config grapes value
    elif index==4:
        orange_value_label.config(text=new_text)   # config orange value
    elif index==5:
        pear_value_label.config(text=new_text)     # config pear value
    elif index==6:
        tomato_value_label.config(text=new_text)   # config tomato value

def img_predict(img):
    # image processing
    img = img.resize((60,60))
    img_arr = np.array(img)
    img_arr = img_arr.reshape((1,) + img_arr.shape)
    img_arr = img_arr.astype('float32')/255
    # make a prediction
    prediction = model.predict(img_arr)
    print(prediction[0])
    # configure bars diagram
    for i in range(1, 7):
        value = prediction[0][i]
        print(my_dict[i]+':', value)
        config_bar(i, value)
        value_acc = 10**display_accuracy
        value_on_screen = str(int(value*value_acc)/value_acc)
        config_label(i, value_on_screen)
    # showing the best prediction
    best_prediction = np.argmax(prediction)
    print('==> Best predicted label:', my_dict[best_prediction])  
    icon_label.config(image=icon_dict[best_prediction])

def update_frame():
    global count
    _, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    img = img.resize((512, 384))
    img_tk = ImageTk.PhotoImage(image=img)
    camera_frame.configure(image=img_tk)
    camera_frame.image = img_tk
    # dự đoán với mô hình CNN và vẽ lại bars
    count+=1
    if count>predict_delay_time:
        count = 0
    if count==predict_delay_time:
        take_photos_automatically()
    # recall this function every 10ms
    camera_frame.after(10, update_frame)


################################################################ Main ################################################################
root = Tk()
root.title("Facial emotional classifier")
window_width = 900
window_height = 399
root.geometry(str(window_width)+'x'+str(window_height))
# camera frame
camera_frame = Label(root)
camera_frame.place(x=5, y=5)
# Create a Canvas bars
gap = 15
max_len = 300
canvas = Canvas(root, width=310, height=250)
canvas.place(x=557, y=30)
bar1 = canvas.create_rectangle(0, 10, max_len, 20, fill='pink', outline='')                 # apple
bar2 = canvas.create_rectangle(0, 40+gap, max_len, 50+gap, fill='orange', outline='')       # banana
bar3 = canvas.create_rectangle(0, 70+2*gap, max_len, 80+2*gap, fill='blue', outline='')     # grapes
bar4 = canvas.create_rectangle(0, 100+3*gap, max_len, 110+3*gap, fill='red', outline='')    # orange
bar5 = canvas.create_rectangle(0, 130+4*gap, max_len, 140+4*gap, fill='green', outline='')  # pear
bar6 = canvas.create_rectangle(0, 160+5*gap, max_len, 170+5*gap, fill='purple', outline='') # tomato
# Load icon
exit_icon = Image.open("icon_img/switch.png")      # initilize power icon
exit_icon = exit_icon.resize((60, 60))
exit_icon = ImageTk.PhotoImage(exit_icon)
camera_icon = Image.open("icon_img/camera.png")    # initilize camera icon
camera_icon = camera_icon.resize((60, 60))
camera_icon = ImageTk.PhotoImage(camera_icon)
auto_icon = Image.open("icon_img/auto.png")        # initilize auto icon
auto_icon = auto_icon.resize((60, 60))
auto_icon = ImageTk.PhotoImage(auto_icon)
manu_icon = Image.open("icon_img/manu.png")        # initilize manual icon
manu_icon = manu_icon.resize((60, 60))
manu_icon = ImageTk.PhotoImage(manu_icon)
apple_icon = Image.open('icon_img/apple.png')      # initilize apple icon
apple_icon = apple_icon.resize((60, 60)) 
apple_icon = ImageTk.PhotoImage(apple_icon)
banana_icon = Image.open('icon_img/banana.png')    # initilize banana icon
banana_icon = banana_icon.resize((60, 60)) 
banana_icon = ImageTk.PhotoImage(banana_icon)
grapes_icon = Image.open('icon_img/grapes.png')    # initilize grapes icon
grapes_icon = grapes_icon.resize((60, 60)) 
grapes_icon = ImageTk.PhotoImage(grapes_icon)
orange_icon = Image.open('icon_img/orange.png')    # initilize orange icon
orange_icon = orange_icon.resize((60, 60)) 
orange_icon = ImageTk.PhotoImage(orange_icon)
pear_icon = Image.open('icon_img/pear.png')        # initilize pear icon
pear_icon = pear_icon.resize((60, 60)) 
pear_icon = ImageTk.PhotoImage(pear_icon)
tomato_icon = Image.open('icon_img/tomato.png')    # initilize tomato icon
tomato_icon = tomato_icon.resize((60, 60)) 
tomato_icon = ImageTk.PhotoImage(tomato_icon)
icon_dict = {1:apple_icon, 2:banana_icon, 3:grapes_icon, 4:orange_icon, 5:pear_icon, 6:tomato_icon}
# Decorating labels
happy_label = Label(text='Apple', font=(None, 12, 'bold'))         
happy_label.place(x=557, y=15)                                         
surprise_label = Label(text='Banana', font=(None, 12, 'bold'))       
surprise_label.place(x=557, y=55)                                      
sad_label = Label(text='Grapes', font=(None, 12, 'bold'))             
sad_label.place(x=557, y=100)                                          
angry_label = Label(text='Oranges', font=(None, 12, 'bold'))             
angry_label.place(x=557, y=145)                                        
disgust_label = Label(text='Pear', font=(None, 12, 'bold'))         
disgust_label.place(x=557, y=190)                                      
fear_label = Label(text='Tomato', font=(None, 12, 'bold'))               
fear_label.place(x=557, y=235)   
# Value labels (configurable)                                
value_x_coordinate = 820               
apple_value_label = Label(text='0.0', font=(None, 10))                 
apple_value_label.place(x=value_x_coordinate, y=18)      
banana_value_label = Label(text='0.0', font=(None, 10))              
banana_value_label.place(x=value_x_coordinate, y=58)    
grapes_value_label = Label(text='0.0', font=(None, 10))                   
grapes_value_label.place(x=value_x_coordinate, y=103)         
orange_value_label = Label(text='0.0', font=(None, 10))                 
orange_value_label.place(x=value_x_coordinate, y=148)         
pear_value_label = Label(text='0.0', font=(None, 10))               
pear_value_label.place(x=value_x_coordinate, y=193)        
tomato_value_label = Label(text='0.0', font=(None, 10))                  
tomato_value_label.place(x=value_x_coordinate, y=238)         
icon_label = Label(root, image=None)                                   
icon_label.place(x=797, y=320)
# button
mode_but = Button(root, image=manu_icon, bd=0, relief="flat", command=switch_mode)
mode_but.place(x=557, y=320, height=60, width=60)
take_photo_but = Button(root, image=camera_icon, bd=0, relief="flat", command=take_photos_manualy)
take_photo_but.place(x=637, y=320, height=60, width=60)
exit_but = Button(root, image=exit_icon, bd=0, relief="flat", command=close_window)
exit_but.place(x=717, y=320, height=60, width=60)
###
update_frame()
root.mainloop()
cap.release()
