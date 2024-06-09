from django.db import models

# Create your models here.


class Messages(models.Model):
    #id = models.AutoField()
    
    user_unique_key = models.CharField(max_length=20, blank=True, null=True, default='')
    user1 = models.TextField(default='')
    user2 = models.CharField(max_length=4000, default='')
    sentiment_analysis=models.CharField(max_length=10, default='')
    timestamp = models.DateTimeField(auto_now_add=True,)
    
    """user1 = models.TextField(default='')    
    user2 = models.CharField(max_length=1000,default='')"""
"""
models.py in Django serves as the place to define the structure of your application's data
. Here's a simplified breakdown:

Data Structure Definition: In models.py, you define the different types of data (called models) that your 
application will work with. Each model typically represents a table in your application's database.
Attributes and Relationships: Within each model, you specify the different attributes (fields) that each 
instance of that model will have. These fields can represent things like strings, numbers, dates, or 
relationships to other models.
Database Interaction: Django uses the information in models.py to automatically generate the necessary
 database tables and relationships. This means you don't have to write SQL code to create tables; 
 Django handles it for you based on your model definitions.
Data Access and Manipulation: Once your models are defined, Django provides an easy-to-use API for interacting
 with your application's data. You can create, read, update, and delete (CRUD) instances of your models using 
 Django's built-in methods and querysets.
Business Logic Separation: Separating data structure definitions from the rest of your code helps keep your
 application organized and maintainable. models.py focuses solely on data structure, while logic for 
 processing and manipulating data typically resides in views.py and other parts of your Django application.
In simpler terms, models.py is where you describe what kind of data your application needs to store, how it's 
structured, and how it relates to other pieces of data. Django then takes care of turning those definitions
 into an actual database, and provides tools for you to work with that data easily in your application's code.
"""    