from django.shortcuts import render, redirect
from .models import *
from django.contrib.auth import authenticate, login, logout
from .forms import *
from django.contrib import messages
from django.contrib.auth.models import Group
from django.contrib.auth.decorators import login_required
from accounts.decorator import *

# Create your views here.

@login_required(login_url='login')
@allowed_users(allowed_roles=['meditator'])
def analysis(request):
    meditation = request.user.profile.meditation_set.all()
    health = request.user.profile.health_set.all()
    m_count = meditation.count()
    h_count = health.count()
    profile = request.user.profile
    context = {'meditation' : meditation, 'health' : health, 'profile' : profile, 'm_count': m_count, 'h_count': h_count}
    return render(request, 'accounts/analysis.html' , context)

@unauthenticated_user
def registerPage(request):
        form = CreateUserForm()

        if request.method == 'POST':
            form = CreateUserForm(request.POST)
            if form.is_valid():
                user = form.save()
                first_name = form.cleaned_data.get('first_name')
                last_name = form.cleaned_data.get('last_name')
                username = form.cleaned_data.get('username')

                group = Group.objects.get(name = 'meditator')
                user.groups.add(group)

                Profile.objects.create(
                    user = user,
                    name = first_name + ' ' + last_name
                )
                messages.success(request, 'Account was created for ' + username)
                return redirect('login')

        context = {'form':form}
        return render(request, 'accounts/register.html', context)



@unauthenticated_user
def loginPage(request):
        if request.method == 'POST':
            username = request.POST.get('username')
            password = request.POST.get('password')

            user = authenticate(request, username = username, password = password)
            if user is not None:
                login(request, user)
                return redirect('index')

            else: 
                messages.info(request, 'Username or password is incorrect')
                return render(request, 'accounts/login.html')

        context = {}
        return render(request, 'accounts/login.html', context)



def logoutUser(request):
    logout(request)
    return redirect('login')