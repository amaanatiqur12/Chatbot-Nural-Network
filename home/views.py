"""

In Django, views.py is a Python module where you define the logic for handling incoming web requests and returning appropriate HTTP responses. Here's what views.py typically does in a Django application:

Request Handling: When a user makes a request to a URL in your Django application, Django's URL dispatcher directs that request to a corresponding view function defined in views.py.
Business Logic: Inside the view function, you can write Python code to perform any necessary business logic. This could involve querying the database, processing form submissions, accessing external APIs, or any other operations needed to fulfill the request.
Data Processing: View functions often retrieve data from the database or other sources, process it as needed, and pass it to templates for rendering. This data processing step prepares the data to be presented to the user in a meaningful way.
Template Rendering: After processing the data, the view function typically renders a template using the render function. This involves combining the template with the processed data to generate HTML content that will be sent back to the user's browser as the response.
Response Generation: Finally, the view function returns an HTTP response object. This could be a full HTML page, a JSON response, a redirect to another URL, or any other type of response appropriate for the request. Django takes care of sending this response back to the user's browser.
Overall, views.py serves as the "controller" in the Model-View-Controller (MVC) architecture of Django applications, handling the logic for processing incoming requests and generating appropriate responses. It's where the core functionality of your web application resides.
"""
from django.contrib.auth.hashers import check_password
from django.shortcuts import render,redirect
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
from .models import Messages
import numpy
import tflearn
import tensorflow
import random
import json
import pickle
from django.http import HttpResponse
from django.contrib.auth.models import User
# Create your views here.
from django.template.loader import get_template
import os
from django.http import HttpResponseRedirect
from django.urls import reverse
from django.contrib.auth.decorators import login_required
from django.contrib import messages

from django.contrib.auth import authenticate, login, logout
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def stemming(content):

    stemmed_content = re.sub('[^a-zA-Z]',' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content
    
def sentiment_analysis():    
    column_names = ["target", "id", "date", "Flag", "user", "text"]
   
    pickle_path = "processed_twitter_data.pkl"

   
    if os.path.exists(pickle_path):
       
        with open(pickle_path, 'rb') as file:
            twitter_data = pickle.load(file)
    else:
        
        twitter_data=pd.read_csv(r"C:\Users\Lenovo\Desktop\GitHub\Chatbot\training.1600000.processed.noemoticon.csv", names=column_names ,encoding="ISO-8859-1")
        twitter_data.replace({'target': {4: 1}}, inplace=True)
        twitter_data['stemmed_content'] = twitter_data['text'].apply(stemming)
        
        with open(pickle_path, 'wb') as file:
            pickle.dump(twitter_data, file)
    X = twitter_data['stemmed_content'].values
    Y = twitter_data['target'].values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
    global vectorizer
    vectorizer = TfidfVectorizer()
    X_train_vector_path = "X_train_vector.pkl"
    X_test_vector_path = "X_test_vector.pkl"
    vectorizer_path = "vectorizer.pkl"
    if os.path.exists(X_train_vector_path) and os.path.exists(X_test_vector_path) and os.path.exists(vectorizer_path):
        
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)
        with open(X_train_vector_path, 'rb') as file:
            X_train = pickle.load(file)
        with open(X_test_vector_path, 'rb') as file:
            X_test = pickle.load(file)
    else:
        
        
        vectorizer = TfidfVectorizer()
        X_train = vectorizer.fit_transform(X_train)
        X_test = vectorizer.transform(X_test)
        
        with open(vectorizer_path, 'wb') as file:
            pickle.dump(vectorizer, file)
        
        with open(X_train_vector_path, 'wb') as file:
            pickle.dump(X_train, file)
        with open(X_test_vector_path, 'wb') as file:
            pickle.dump(X_test, file)
    global model2
    Model_path = "Model.pkl"                
    if os.path.exists(Model_path):
        with open(Model_path, 'rb') as file:
            
            model2 = pickle.load(file)
    else:
         
        model2 = LogisticRegression(max_iter=1000)    
        model2.fit(X_train, Y_train)
        with open(Model_path, 'wb') as file:
            pickle.dump(model, file)
    

def sentiment_analysis_users(User_chat):
    str1=stemming(User_chat)
    X_test1 = vectorizer.transform([str1]) 
    prediction = model2.predict(X_test1)

    
    if prediction[0] == 1:
        data = "positive" 
    else:
        data = "negative"
    return data


@login_required(login_url="/login/")
def chat_view(request):
    if request.method == 'POST':
        
        user1 = request.POST.get('text')
        result_sentiment=sentiment_analysis_users(user1)
        
        if user1.strip():  
            message = Messages(user1=user1, user2=user2, user_unique_key=request.user.id, sentiment_analysis=result_sentiment) 
            message.save()
            
         
        return redirect("http://127.0.0.1:8000/")

    
    messages = Messages.objects.filter(user_unique_key=request.user.id)

    
    return render(request, 'home.html', {'messages': messages})
"""
Yes, exactly! When a user visits the URL corresponding to the 'home.html' template in the web application,
 this line of code is responsible for generating the content that will be displayed to the user in their web
  browser.

Here's how it works:

The user's browser sends a request to the Django server for the 'home.html' page.
Django receives the request and executes the view function associated with that URL.
Within the view function, the render function is called, instructing Django to render the 'home.html' template.
Along with rendering the template, the render function also passes along any necessary data. In this case, it's
 passing a dictionary with the key 'messages' containing the messages that should be displayed on the 'home.html'
  page.
Finally, Django sends the rendered HTML content, along with any associated data, back to the user's browser as
 the response to their request. The browser then displays this content to the user.
So yes, you're correct. When a user goes to the 'home.html' page, this line of code triggers the rendering of
 that page with the associated data.
"""

def MLCode():
    global model

    template_path = get_template('intents.json').origin.name
    
    
    directory = os.path.dirname(template_path)
    
    intents_json_path = os.path.join(directory, 'intents.json')
    print(intents_json_path)

    

    with open(intents_json_path) as file:
        global data
        data = json.load(file)
    
    
        
    try:
       
        script_dir = os.path.dirname(__file__)
        chatbot_dir = os.path.abspath(os.path.join(script_dir, '..'))
        
        data_pickle_path = os.path.join(chatbot_dir, 'data.pickle')
        
        """template_path = get_template('data.pickle').origin.name
        print(template_path)
        directory = os.path.dirname(template_path)
        print(directory)
        intents_json_path = "C:/Users/Lenovo/Desktop/chatbot/data.pickle"""
        with open(data_pickle_path, "rb") as f:
            print("I in the loop")
            global words
            global labels
            global training
            global output
            words, labels, training, output = pickle.load(f)
          
    except:
        words = []
        labels = []
        docs_x = []
        docs_y = []

        for intent in data["intents"]:
            for pattern in intent["patterns"]:
           
                wrds = nltk.word_tokenize(pattern)
                words.extend(wrds)
                docs_x.append(wrds)
                docs_y.append(intent["tag"])

            if intent["tag"] not in labels:
                labels.append(intent["tag"])
      

        words = [stemmer.stem(w.lower()) for w in words if w != "?"]
        words = sorted(list(set(words)))
       

        labels = sorted(labels)

        training = []
        output = []

        out_empty = [0 for _ in range(len(labels))]
    

        for x, doc in enumerate(docs_x):
            bag = []
            
            wrds = [stemmer.stem(w.lower()) for w in doc]
    
            for w in words:
                if w in wrds:
                    bag.append(1)
                else:
                    bag.append(0)

            output_row = out_empty[:]
            output_row[labels.index(docs_y[x])] = 1
       
            training.append(bag)
            output.append(output_row)

  
    
        training = numpy.array(training)
        output = numpy.array(output)

        with open("data.pickle", "wb") as f:
            pickle.dump((words, labels, training, output), f)
    tensorflow.compat.v1.reset_default_graph()

    net = tflearn.input_data(shape=[None, len(training[0])])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
    net = tflearn.regression(net)

    model=tflearn.DNN(net)


    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

global port_stem 
port_stem = PorterStemmer()
sentiment_analysis()   
MLCode()




    
    
   
    
    
    
    

@login_required(login_url="/login/")
def chat(request):
    
    

    
    
    if request.method =="POST":
        
        
        while True:
            
            inp = request.POST.get('text')
           
            if inp.lower() == "quit":
                break
            
            results = model.predict([bag_of_words(inp, words)])[0]
            results_index = numpy.argmax(results)
            tag = labels[results_index]
            
            if results[results_index] > 0.7:

                for tg in data["intents"]:
                     if tg['tag'] == tag:
                        responses = tg['responses']

                
                global user2
                user2 = random.choice(responses)

                return chat_view(request)
                
                
            else:
                global users2
                user2 = "I didn't get that try again"
                return chat_view(request)
                
                
    messages = Messages.objects.filter(user_unique_key=request.user.id)
   
    
    return render(request, 'home.html', {'messages': messages})            

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)

def logout_page(request):
    logout(request)
    return redirect('/login/')

def register(request):
    if request.method =="POST":
        
        first_name =request.POST.get('first_name')
        last_name =request.POST.get('last_name')
        username =request.POST.get('username')
        password =request.POST.get('password')

       
        user = User.objects.filter(username=username)

        if user.exists():
            messages.info(request, "Username already taken")
            return redirect('/register/')

        user = User.objects.create(
            first_name  = first_name,
            last_name  =  last_name,
            username  = username,
            
            )
      
        
        user.set_password(password)
        user.save()
        messages.error(request, "Account created successfully")
        return redirect('/register/')

    return render(request, 'register.html')

def login_page(request):
    if request.method =="POST":
        

      
    
    
        username =request.POST.get('username')
        password =request.POST.get('password')
        
        """try:
            user = User.objects.get(username=username)
        except User.DoesNotExist:
            messages.error(request, "Invalid Username")
            return redirect('/login/')"""    

        if not User.objects.filter(username=username).exists():
            messages.error(request, "Invalid Username")
            return redirect('/login/')   
       
        user = authenticate( username = username , password = password )
        
        if user==None:
            messages.error(request, "Invalid password")
            return redirect('/login/')
        else:
            login(request , user)
            return redirect("http://127.0.0.1:8000/")
        """print(password,"Amaan")
        print(user.password)    
        if check_password(password, user.password):
            # Password is correct
            login(request, user)
            return redirect("http://127.0.0.1:8000/")
        else:
            # Password is incorrect
            messages.error(request, "Invalid password")
            return redirect('/login/')"""    

    return render(request, 'login.html') 