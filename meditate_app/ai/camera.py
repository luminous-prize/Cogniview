import cv2 # OpenCV library for computer vision, machine learning, and image processing 
import os # OS Module provides functions for interacting with the operating system
from keras.models import load_model # Used to load the saved trained model 
import numpy as np # Used to perform a wide variety of mathematical operations on arrays
from pygame import mixer # Used for loading and playing sounds
import time # Used to handle time
import mediapipe as mp # MediaPipe is a ML framework developed by Google. MediaPipe has various pre-trained models inbuilt.
from imutils.video import VideoStream # Used to capture the live video from webcam. We chose it over OpenCV's VideoCapture as imutils uses threads which in return runs faster compared to OpenCV
import imutils # Used to make basic image processing function calls
from django.conf import settings 
from med_app.settings import BASE_DIR, STATIC_URL

mixer.init() # Initializes the mixer module

sound = mixer.Sound(BASE_DIR + '\\ai\static\\ai\media\warning.mp3') # Loading the warning sound
end_sound = mixer.Sound(BASE_DIR + '\\ai\static\\ai\media\end.mp3') # Loading the end message sound

leye = cv2.CascadeClassifier(BASE_DIR + '\\ai\static\\ai\\files\haarcascade_lefteye_2splits.xml') # Loading the HaarCascade Left Eye Classifier
reye = cv2.CascadeClassifier(BASE_DIR + '\\ai\static\\ai\\files\haarcascade_righteye_2splits.xml') # Loading the HaarCascade Right Eye Classifier

model = load_model(BASE_DIR + '\\ai\static\\ai\\files\cnnCat2.h5') # Loading the saved trained model for detecting whether the eyes are open or not
path = os.getcwd() # Storing the current working directory
font = cv2.FONT_HERSHEY_COMPLEX_SMALL # Loading the Hershey Complex Small Font which is to be used to display the messages on screen

image_path = BASE_DIR + '\\ai\static\\ai\images\zen_bg.jpg' # Storing the path to the Zen Background image

# MediaPipe "Selfie Segmentation" segments the prominent humans in the scene. 
# It can run in real-time which can be used for our purpose of detecting the human and changing the background

mp_selfie_segmentation = mp.solutions.selfie_segmentation # Initializing the selfie segmentation model
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1) # We chose model 1 which is a landscape model rather than model 0 which is a general model
# as model 1 runs raster as it has fewer FLOPs than the general model (FLOPS - refers to the number of floating point operations that can be performed by a computing entity in one second)

class VideoCamera(): # Creating a video camera class which will be used for streaming the real-time video captured which would detect whether the person is meditating or not; and also change the background while meditating
    def __init__(self, TIME, blinked): # Initializing the default constructor
        self.vs = VideoStream(src=0).start() # Initialzing the VideoStream object with the source as our web cam
        self.prev = time.time() # Used to get the time
        self.TIMER = TIME # Setting the timer for the meditation (argument is an integer which is considered as seconds)
        self.finish = False # Setting a flag to trigger the completion of meditation
        self.blinked = blinked
        
    def __del__(self): # Function to destroy all windows
        cv2.destroyAllWindows() # Used to close all the open windows created by OpenCV to avoid memory leak after running the script 

    def get_frame(self, rpred, lpred, lbl, lbl_pred): # Main Logic of the script
        frame = self.vs.read() # Read the frame capture by the webcam
        frame = imutils.resize(frame, width=650) # Resize the frame
        height , width, _ = frame.shape # Getting the height and widht of the frame
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Converting the color of the frame to grayscale which will be used to detect the face and eyes using Haar Cascade
        RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Converting the color of the frame to RGB which will be for segmentation using MediaPipes selfie_segmentation
        left_eye = leye.detectMultiScale(gray) # Using the Left Eye Classifier we will detect the left eye in the frame
        right_eye =  reye.detectMultiScale(gray) # Using the Right Eye Classifier we will detect the right eye in the frame

        results = selfie_segmentation.process(RGB) # Performs the segmentation process and return a probability map with pixel values near 1 for the indexes where the person is located in the image and pixel values near 0 for the background.
        mask = results.segmentation_mask # Extracts the mask from the result class
        condition = np.stack( # np.stack joins a sequence of arrays along a new axis.
        (mask,) * 3, axis=-1) > 0.6 # Returns a matrix of shape as the mask. It contains true where the pixel value is more than 0.6 and returns false where the pixel value is less than 0.6

        bg_image = cv2.imread(image_path) # Reads the Zen background image
        bg_image = cv2.resize(bg_image, (width, height)) # Resize it to the same size as the frame


        for (x,y,w,h) in right_eye: # Iterating over the detected right eye with its x,y coordinates 
            r_eye = frame[y:y+h,x:x+w] # Slicing the frame array to get the right eye
            r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY) # Converting the color of the eye to grayscale
            r_eye = cv2.resize(r_eye,(24,24)) # Resizing the eye frame to 24 x 24 as the model to detect eyes was trained on a 24 x 24 input ;_;
            r_eye = r_eye/255 # Normalizing the eye array (we divide it by 255 as the pixel values range from 0 to 256, apart from 0 the range is 255. So dividing all the values by 255 will convert it to range from 0 to 1)
            r_eye =  r_eye.reshape(24,24,-1) #  Reshape the array to 24 x 24 x -1 (in the reshape method it reshapes the array without changing the original size, so for the 3rd dimension we use -1 as it will adjust to the original size)
            r_eye = np.expand_dims(r_eye,axis=0) # Adds a dimension to the array at the 0th axis. The extra dimension at the beginning of images are generally for batch sizes. Batch size represent how many images you bundle to feed the CNN at once
            rpred = model.predict(r_eye) # Here, r_eye is the input data for the model to predict on and it returns the probabilty of each class
            rpred = np.argmax(rpred,axis=1) # We set axis = 1 , so that argmax identifies the maximum value for every row. And it returns the column index of that maximum value.
            if(rpred[0]==1): 
                lbl_pred=lbl[1]    #'Distracted' 
            if(rpred[0]==0):
                lbl_pred=lbl[0]    #'Meditating'
            break # Break out of the iteration

        for (x,y,w,h) in left_eye: # Iterating over the detected left eye with its x,y coordinates 
            l_eye = frame[y:y+h,x:x+w] # Slicing the frame array to get the left eye
            l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY) # Converting the color of the eye to grayscale
            l_eye = cv2.resize(l_eye,(24,24)) # Resizing the eye frame to 24 x 24 as the model to detect eyes was trained on a 24 x 24 input ;_;
            l_eye = l_eye/255 # Normalizing the eye array (we divide it by 255 as the pixel values range from 0 to 256, apart from 0 the range is 255. So dividing all the values by 255 will convert it to range from 0 to 1)
            l_eye = l_eye.reshape(24,24,-1) #  Reshape the array to 24 x 24 x -1 (in the reshape method it reshapes the array without changing the original size, so for the 3rd dimension we use -1 as it will adjust to the original size)
            l_eye = np.expand_dims(l_eye,axis=0) # Adds a dimension to the array at the 0th axis. The extra dimension at the beginning of images are generally for batch sizes. Batch size represent how many images you bundle to feed the CNN at once
            lpred = model.predict(l_eye) # Here, l_eye is the input data for the model to predict on and it returns the probabilty of each class
            lpred = np.argmax(lpred,axis=1) # We set axis = 1 , so that argmax identifies the maximum value for every row. And it returns the column index of that maximum value.
            if(lpred[0]==1):
                lbl_pred=lbl[1]    #'Distracted'   
            if(lpred[0]==0):
                lbl_pred=lbl[0]    #'Meditating'
            break # Break out of the iteration

        cur = time.time() # Reading current time to check if the score has increased after the eye was open

        prev_time = self.TIMER # Time in the previous iteration

        if(rpred[0]==1 and lpred[0]==1): # If the eyes are open
            if self.TIMER>=0: # To check if the timer isn't over
                if cur-self.prev >= 1:  # To check if the meditation has been started after calling the object, i.e, prev is the time when the meditation started and cur is the current time at which it is checking the state of meditation after the prediction
                    self.prev = cur # Set the previous timer to the current time
                    self.TIMER += 1 # Increase the timer by 1 second
                    self.blinked += 1
            cv2.rectangle(frame, (0,height-50) , (150,height) , (0,0,255) , thickness=cv2.FILLED ) # Make a red rectangle at bottom left to indicate the distracted state
            cv2.putText(frame,lbl_pred,(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA) # Text on the red rectangle which reads "Distracted"

        else: # If the eyes are closed
            if self.TIMER>=0: # To check if the timer isn't over
                if cur-self.prev >= 1: # To check if the meditation has been started after calling the object, i.e, prev is the time when the meditation started and cur is the current time at which it is checking the state of meditation after the prediction
                    self.prev = cur # Set the previous timer to the current time
                    self.TIMER = self.TIMER-1 # Decrease the timer by 1 second
            cv2.rectangle(frame, (0,height-50) , (150,height) , (0,255,0) , thickness=cv2.FILLED ) # Make a green rectangle at bottom left to indicate the meditative state
            cv2.putText(frame,lbl_pred,(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA) # Text on the green rectangle which reads "Meditating"

            # combine frame and background image using the condition
            frame = np.where(condition, frame, bg_image)
        
        cv2.putText(frame,'Meditation Meter:'+str(self.TIMER),(150,height-20), font, 1,(255,255,255),1,cv2.LINE_AA) # Text denoting the TIMER of the meditation

        if(self.TIMER==1): # If the timer of the meditation reaches 1, it turns the screen black
            frame=np.zeros([height,width,3], np.uint8) # Setting the frame to a black screen

        if(self.TIMER<1): # When the timer hits 0, i.e, meditation is over
            self.TIMER = 0 # Set the timer to 0
            self.finish = True # Setting the flag to True to denote that the meditation is over
            try: # Exception Handling - Try to do this
                end_sound.play() # Plays the 'meditation is over' audio
            except: # If the try statement fails, we come to the 'except' statement
                pass # Pass and move to the next line of code

        if(self.TIMER>prev_time and self.TIMER%4==0): # It checks if the TIMER is greater than the previous time and is divisible by 4 (We do this divisibilty check as the audio plays every second that the person is distracted and it overlaps over each other and creates an disturbing sound)
            # If the person is distracted, we set off the warning message
            try: # Exception Handling - Try to do this
                sound.play() # Plays the 'please close your eyes' audio
                
            except: # If the try statement fails, we come to the 'except' statement
                pass # Pass and move to the next line of code

        _, jpeg = cv2.imencode('.jpg', frame) # As the HttpResponse of an video frame requires it to be in the jpg format, so we convert (encode) image format into streaming data (i.e, jpg)
        return jpeg.tobytes(), self.finish, self.blinked # Return the jpg and the finish flag

