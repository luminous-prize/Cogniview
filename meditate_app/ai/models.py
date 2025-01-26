from __future__ import unicode_literals

# Django web applications access and manage data through Python objects referred to as models. 
# Models define the structure of stored data, including the field types and possibly also their maximum size, 
# default values, selection list options, help text for documentation, label text for forms, etc
from django.db import models

class Audio_store(models.Model): # A model class to store the uploaded audio which will be used to classify the emotions from it
    record=models.FileField(upload_to='documents/') # A FileField to get the uploaded file and uploading it to the documents folder which automatically gets created in the media folder (this is by default in django for uploading any file)
    class Meta: 
        db_table='Audio_store' # Creating a database to store the audio files


class Audio_store(models.Model): # A model class to store the uploaded audio which will be used to classify the emotions from it
    record=models.IntegerField() # A FileField to get the uploaded file and uploading it to the documents folder which automatically gets created in the media folder (this is by default in django for uploading any file)
    class Meta: 
        db_table='Time_store'

