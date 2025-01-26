import math # Maths Library
import os # OS Module provides functions for interacting with the operating system
import fleep # Determines file format by file signature
#from pydub import AudioSegment # Used to work with audio files (didn't use mixer library as it can only play audio, whereas pydub can manipulate the audio)
import pydub # Used to work with audio files
import pickle # Used to read the labels file
import pandas as pd # Used to create DataFrames
import numpy as np # Used to perform a wide variety of mathematical operations on arrays
from keras.models import model_from_json # Used to load the saved trained model from json
import tensorflow as tf # An end-to-end platform for machine learning 
import librosa # Used for audio analysis; and used to convert audio into MFCC
from med_app.settings import BASE_DIR 
pydub.AudioSegment.ffmpeg = "C:/ffmpeg/bin/ffmpeg.exe" # Used to execute the ffmpeg file for pydub library


class SplitWavAudioMubin(): # Created a class to split the given audio
    def __init__(self, folder, filename): # Initializing the default constructor
        self.folder = folder # Get the folder directory of the file
        self.filename = filename # Get the filename
        self.filepath = folder + '/' + filename # Get the complete path of the file 
        
        self.audio = pydub.AudioSegment.from_wav(self.filepath) # Read the audio from the path of the audio in milliseconds
    
    def get_duration(self): # To get the duration of the audio file
        return self.audio.duration_seconds # Returns the duration of the audio in seconds
     
    def single_split(self, from_t, to_t, split_filename): # Split audio into a single segment from t1 to t2 and rename it with new filename
        t1 = from_t * 1000 # Converting the from_t to second
        t2 = to_t * 1000 # Converting the to_t to second
        split_audio = self.audio[t1:t2] # Slice the audio file from 'from_t' seconds to 'to_t' seconds as AudioSegments are slicable
        split_audio.export(self.folder + '/' + split_filename, format="wav") # Export the audio file to the folder with a new name in 'wav' format
        
    def multiple_split(self,segment_length): # Split audio into multiple segments of a particular segment length
        total_t = math.ceil(self.get_duration()) # Rounds the duration UP to the nearest integer, if necessary, and returns the total duration
        for i in range(0, total_t,segment_length): # Iterating from to total_t with a step size of the segment_length
            split_fn = str(i) + '_' + self.filename # Setting the new filename as the 'i' values + '_' + the original filename
            self.single_split(i, i+segment_length, split_fn) # Calling the split audio function
        return(i,total_t) # Returns the total number of segments in which the audio gets splited

def get_audio(k,folder,file): # Created a class to get the audio file, check the format, and split the audio
  if str(os.path.isdir(folder)) =='True' : # Check if the directory exists
    f=0 # Flag to check if the audio is correctly formatted or not
    while (f==0): # While the Flag is set off
      if str(os.path.isfile(folder + file)) == 'True': # Check if the file exists in that directory
        with open(folder + file, "rb") as fi: # Opening the audio file as 'fi'
          info = fleep.get(fi.read(128)) # Determines the file extension
        if info.extension[0] == 'mp3' : # Check if the file extension is 'mp3'
          src = folder + file # File path
          dst = src + '.wav' # File path with extension as '.wav'
          # convert mp3 to wav
          sound = pydub.AudioSegment.from_mp3(src) # Read the mp3 file
          sound.export(dst, format="wav") # Convert the mp3 into wav file using the dst given
          file=file+'.wav' # Reading the converted wav file with extension
          split_wav = SplitWavAudioMubin(folder, file) # Creating an object of the 'SplitWavAudioMubin' class
          k,duration = split_wav.multiple_split(segment_length=5) # Split the audio in multiple segments of 5 secs
          f=1 # Setting the conversion flag to 1 so that it breaks out of the function call
        elif info.extension[0] == 'm4a' or info.extension[0] == 'wav': # Check if the file extension is 'm4a' or 'wav'
          split_wav = SplitWavAudioMubin(folder, file) # Creating an object of the 'SplitWavAudioMubin' class
          k,duration = split_wav.multiple_split(segment_length=5) # Split the audio in multiple segments of 5 secs
          f=1 # Setting the conversion flag to 1 so that it breaks out of the function call
        else: # If the chosen audio file doesn't match the extension 'mp3, 'm4a', 'wav'
          print("Please enter name of audio file with supported a format (mp3,m4a,wav)") # Prints a warning message to give an indication to upload file in these 3 given format
      else: # If the file doesn't exist at the given directory
        print("Please ensure that the file is in the directory specified above and verify the file name") # Prints a warning message to choose the correct file directory with correct file name
        break # Break out of the loop
    return(k, file, duration) # Returns the number of segments and the file
  else: # If the directory doesn't exists
    print('Check if directory is correct') # Prints a warning message to check if the directory is correct or not
    get_audio() # Calls the function again 

def app(k,folder,file): # Function which is the main logic of the prediction

    k, file, duration = get_audio(0,folder,file) # Call the get_audio file with the audio split into segments of 5 secs
    n = k/5 # Gets the number of segments
    ans = {} # Store the predictiosn of each segments
    json_file = open(BASE_DIR + '\\ai\static\\ai\\files\model_json.json', 'r') # Open the saved trained model
    loaded_model_json = json_file.read() # Read the model
    json_file.close() # Close the json file of the model

    loaded_model = model_from_json(loaded_model_json) # Read the model for prediction
    loaded_model.load_weights(BASE_DIR + '\\ai\static\\ai\\files\Emotion_Model.h5') # Load the weights
    print("Model Loaded!") # Confirms that the model has been loaded

    opt = tf.keras.optimizers.RMSprop(lr=0.00001, decay=1e-6) # Used to speed up gradient descent so that it can converge faster to the minima. 
    loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy']) # Compile the model 
    files = [file] # Maintain a list of the main audio file along with the segments

    for i in range(int(n)): # Iterating from 0 to the number of segments
        # Transform Dataset

        #Load the audio file segment as a floating point time series. Audio will be automatically resampled to the given rate; with a duration of 2.5 secs and an offset of 0.5 secs; and resample type as 'kaiser_fast' which helps in loading the audio file faster
        X, sample_rate = librosa.load(folder + str(5*i) + '_' + file,res_type='kaiser_fast',duration=2.5,sr=44100,offset=0.5) 
        files.append(str(5*i) + '_' + file) # Append the audio segment file name to the files list
        sample_rate = np.array(sample_rate) # Converting the sample rate into a numpy array
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13),axis=0) # Taking the mean of 13 mfcc calculated on the audio time series 'X' with a 'sample_rate' sample rate
        newdf = pd.DataFrame(data=mfccs).T # Taking the transpose of the DataFrame created with the data as the mfccs

        # Predictions
        newdf= np.expand_dims(newdf, axis=2) # Adds a dimension to the array at the 2nd axis
        newpred = loaded_model.predict(newdf, 
                                batch_size=16, 
                                verbose=1) # Here newdf is the input data for the model to predict on and it returns the probabilty of each class with batch_size of 16

        filename = BASE_DIR + '\\ai\static\\ai\\files\labels' # Directory of the labels file
        infile = open(filename,'rb') # Open the labels file

        lb = pickle.load(infile) # Loading the labels
        infile.close() # Close the labels file

        # Final Prediction
        final = newpred.argmax(axis=1) # We set axis = 1 , so that argmax identifies the maximum value for every row. And it returns the column index of that maximum value.
        final = final.astype(int).flatten() # Flatten the array into one dimension and cast the values into an integer
        final = (lb.inverse_transform((final))) # Transforming label back to original encoding
        print(f"FOR {i} emotion is {final[0]}") # Prints the emotion for the current segment (For eg: FOR 0 emotion is female_angry)
        ans[(i+1)*5] = final[0] # Adding a new element to the predictions dictionary with key as the segments of 5 and value as the emotion detected in that 5 secs
        print('Predicted label:',final) # Prints the predicted label
    
    try: # Exception Handling - Try to do this
      files.append(str(5*(i+1)) + '_' + file) # Adding the last segment to the files list
      files.append(file.replace('.wav','')) # Adding the file name without the extension
    except: # If the try statement fails, we come to the 'except' statement
      files.append(file.replace('.wav','')) # Adding the file name without the extension

    # Deleting the files after prediction to free up storage

    if n==0: # If only one or no segments are formed
      if os.path.exists(folder+'0_'+file): # Check if the file exists
        os.remove(folder+'0_'+file)  # Remove the file 

    for i in files: # Iterate through the files list
      if os.path.exists(folder+i): # Check if the file exists
        os.remove(folder+i) # Remove the file

    return ans,duration # Return the predictions dictionary

    