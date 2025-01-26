from django import forms # The audio file is taken as an input to be uploaded via a form
from ai.models import * # Importing the Audio_store model to handle the audio data that will be uploaded through the form

class AudioForm(forms.ModelForm): # Create a form called AudioForm
    class Meta:
        model=Audio_store # Using the Audio_store model
        fields=['record'] # Store the uploaded file into the 'record' field of the database that will get created from the model


## NEED A FORM TO GET THE DURATION FOR MEDITATION AND SET IT AS THE TIME FOR THE FUNCTION CALL

class DurationForm(forms.Form): 
    class Meta:
        model=Audio_store 
        fields=['record']