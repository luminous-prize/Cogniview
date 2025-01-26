from django.db import models
from django.contrib.auth.models import User
from django.utils.timezone import now
# Create your models here.

class Profile(models.Model):
    user = models.OneToOneField(User, null= True, on_delete=models.CASCADE)
    name = models.CharField(max_length=200, null= True)
    date_created = models.DateTimeField(auto_now_add=True, null= True)

    def __str__(self):
        return self.name


class Meditation(models.Model):
    profile = models.ForeignKey(Profile, null= True,on_delete=models.SET_NULL)
    date_created = models.DateTimeField(auto_now_add=True, null= True)
    duration = models.PositiveIntegerField(null= True)
    no_distracted = models.PositiveSmallIntegerField(null= True)
    score = models.PositiveIntegerField(null= True)


class Health(models.Model):
    profile = models.ForeignKey(Profile,null= True, on_delete=models.SET_NULL)
    date_created = models.DateTimeField(auto_now_add=True,null= True)
    duration = models.PositiveIntegerField(null= True)
    emotions = models.CharField(max_length=200,null= True)
    gender = models.CharField(max_length=6,null= True)
