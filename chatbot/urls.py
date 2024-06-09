"""
URL configuration for chatbot project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from home.views import *

urlpatterns = [
    path("admin/", admin.site.urls),
    path('' , chat , name="chat"),
    path('ml-code/', MLCode, name='ml_code'),
    path('chat_view/', chat_view, name='chat_view'),
    path('register/' , register , name="register"),
    path('login/' , login_page , name="login_page"),
    path('logout/' , logout_page , name="logout_page"),
    path('sentiment-analysis/', sentiment_analysis, name='sentiment_analysis'),
]
"""urls.py is a file in Django projects where you define the mapping between URLs and views. It tells
 Django which view function to call when a specific URL is requested by a user. So, it's like a roadmap
  that directs web traffic to different parts of your Django application based on the requested URL.

  When I mention "URL pattern as a string," I mean the part of the URL that comes after the domain name. 
  For example, if your website domain is example.com, then the URL pattern in the context of urls.py would 
  be everything after example.com.

Let's break it down with an example:

Suppose you have a Django website with the domain example.com, and you want to create a URL pattern for a 
page that shows a user's profile. In your urls.py file, you might have something like this:

python
Copy code
urlpatterns = [
    path('profile/', views.profile_view, name='profile'),
]
In this example:

The URL pattern as a string is 'profile/'.
It means that when a user visits example.com/profile/, Django will call the profile_view function to handle 
that request.
So, the URL pattern 'profile/' corresponds to the /profile/ part of the URL. It's what Django matches
against the incoming URL to determine which view function to call.


The second argument in a Django path() function is the view function that will be executed when the 
corresponding URL pattern is matched.

In Django, a view function is a Python function that receives an HTTP request and returns an HTTP response.
 It contains the logic for processing the request and generating the response.

When a URL is requested by a user, Django's URL dispatcher matches the requested URL pattern to the ones 
defined in your urls.py file. Once a match is found, Django calls the associated view function to handle
 the request.

For example, consider the following urls.py snippet:

python
Copy code
from django.urls import path
from . import views

urlpatterns = [
    path('about/', views.about_view, name='about'),
]
In this snippet:

The second argument views.about_view is the view function that will be called when the URL pattern 'about/' 
is matched.
views.about_view refers to a Python function named about_view defined in a module named views (which is 
imported with from . import views).
When a user visits the URL /about/, Django will call the about_view function to handle the request, executing 
its code and generating the appropriate response to send back to the user's browser


The third argument in a Django path() function is optional, and it's the name of the URL pattern. This name
 can be used for reverse URL lookup, which is a way to generate URLs dynamically in Django templates or Python 
 code without hardcoding them.

Let's break it down with an example:

Suppose you have the following URL pattern defined in your urls.py file:

python
Copy code
from django.urls import path
from . import views

urlpatterns = [
    path('about/', views.about_view, name='about'),
]
In this case:

The third argument 'about' is the name of the URL pattern.
This allows you to refer to this specific URL pattern by its name, which is 'about'.
Now, in your Django templates or Python code, you can use this name to dynamically generate URLs using the 
url template tag or the reverse() function.

For example, in a Django template:

html
Copy code
<a href="{% url 'about' %}">About Us</a>
In this template code, {% url 'about' %} dynamically generates the URL for the 'about' URL pattern 
defined in your urls.py file.

Similarly, in Python code, you can use the reverse() function:

python
Copy code
from django.urls import reverse

url = reverse('about')
This will dynamically generate the URL for the 'about' URL pattern, which you can use in your Python code.
 Using names for URL patterns makes your code more maintainable and avoids hardcoding URLs, which can be 
 error-prone and harder to maintain.
"""