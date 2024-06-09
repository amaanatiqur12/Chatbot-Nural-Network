----------------------------------- Chatbot neural network---------------------------------------------
 

import json
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random


"""Saving intents.json file value to data
 Below is the example of intents.jason data 
{  "intents": [   {
            "tag": "greeting",
            "patterns": [   "Hi",   "Hey",  ],
            "responses": [   "Hey there!", "Hello! How can I assist you?",] 
             },  ]  }"""
with open('intents.json') as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:

    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data['intents']:
        for pattern in intent['patterns']:

        
""" For Example "patterns": ["Hi", "How are you", "Is anyone there?", "Hello", "Good day"], => This is the pattern
This is the result after extends => ['Hi', 'How', 'are', 'you', 'Is', 'anyone', 'there', '?', 'Hello', 'Good', 'day']
It is using extend
It will tokenize Each Sentence
The line wrds = nltk.word_tokenize(pattern) is using the nltk.word_tokenize() function to tokenize the words
in the given pattern string. Tokenization is the process of breaking down a text into individual words or 
tokens.Yes, you can use the split() method to split a string into words as an alternative to
nltk.word_tokenize(). The key difference lies in how they handle punctuation and other non- characters. """
            
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)



""" Saving wrds values one by one by using append
For Example patterns: ["Hi", "How are you", "Is anyone there?", "Hello", "Good day"],
[['Hi'], ['How', 'are', 'you'], ['Is', 'anyone', 'there', '?'], ['Hello'], ['Good', 'day'] """
            docs_x.append(wrds)
            


            
 """ Saving tag values one by one, to make a trace which pattern belong to which tag (docs_x(pattern) <=> docs_y(tag))
 => It means the 3 index of docs_x will belong to xyz label of docs_y """
            docs_y.append(intent["tag"])
            


            
# labels variable store all the tag values in the form of List
        if intent['tag'] not in labels:
            labels.append(intent['tag'])
            



"""Stemming reduces words to their root form to group similar words together,
for example, "running" becomes "run," while "ran" and "runner" remain unchanged.
Stemming =>  Use fixed rules such as remove able, ing etc. to derive a base word
ability =>  abil
Lemmatization =>   Use knowledge of a language (a.k.a. linguistic knowledge) to derive a base word
ability => ability"""
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    
    
    
# To remove the duplicate values
    words = sorted(list(set(words)))

    
#As there is no duplicate value, We are only sorting the labels(tag)
    labels = sorted(labels)


    training = []
    output = []
    out_empty = [0 for _ in range(len(labels))]


""" Here in Docs_x each sub array is ilterated and first it get through stemming and then it will check by words with 
each value of words to each sub arrays of docs_x.
    
And then it ilterated with 1 and 0 in same length values of words and appends in training 
For tags it check which tags it there through docs_y and find the values in lables and appends it with value 1 """
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
    

"""The line tensorflow.reset_default_graph() is used to clear the default graph stack and reset the global default graph in TensorFlow.
Here's a breakdown of what it means and why it's used:

TensorFlow Graphs:
 In TensorFlow, all operations (like creating variables, defining the model, etc.) are added to a computational graph.
 The default graph is a global instance of a computational graph to which operations are added.
 Why Reset the Graph?:
    
When you build and train multiple models in the same script or notebook, the default graph can accumulate nodes from previous models.
This can lead to unexpected behavior, errors, or increased memory usage because the old nodes are still present in  the graph.
Purpose of tensorflow.reset_default_graph():
It resets the global default graph to a new, empty state.
This ensures that any previous operations and nodes are cleared out, and you start with a fresh graph.
It is particularly useful in an interactive environment like Jupyter notebooks where the script might be run multiple times.
In summary, tensorflow.reset_default_graph() is used to avoid conflicts and ensure a clean state when defining a new
TensorFlow model. This is important for avoiding issues related to reusing a computational graph that might still
contain operations from previous model definitions."""
tensorflow.reset_default_graph()





#This is input layer with nodes equal to length of words(number of words in list of words)
net = tflearn.input_data(shape=[None, len(training[0])])

#Hidden layer
net = tflearn.fully_connected(net, 8)
#Hidden Layer
net = tflearn.fully_connected(net, 8)

"""Output layer with number of nodes equal to length of Label. And softmax which describe the probability of words in each
ouput nodes"""
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")

"""In summary, the line net = tflearn.regression(net) integrates the regression layer, which finalizes the training setup
for your neural network. This is crucial for the training process, as it defines how the model learns from the data by
minimizing the loss
  
The tflearn.regression layer is essential because it sets up the loss function, optimizer, and training configuration 
for your model. """  
net = tflearn.regression(net)

"""The line model = tflearn.DNN(net) is crucial as it creates an instance of the deep neural network using the defined
 architecture. This model instance provides methods to train, evaluate, and make predictions with the neural network, 
 encapsulating all the necessary configurations for practical usage."""
model = tflearn.DNN(net)






"""we will fit our data to the model to train it. The number of epochs we set is the amount of times that the model will 
see the same information while training."""
try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")



#Same concept as bag of Words(Counting each words and for this it will be one(if it occur twice but we consider it as one))
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)



#It will predict(which tag it belong) and will choise randomly from the tags the response(only one).
def chat():
    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        print(random.choice(responses))

chat()


--------------------------------TEXT SENTIMENT ANAYLSIS------------------------------------------------


def stemming(content):
    #It remove all the things except alphabet
    stemmed_content = re.sub('[^a-zA-Z]',' ', content)
    #Convert all the words in lower case
    stemmed_content = stemmed_content.lower()
    #Split the sentence
    stemmed_content = stemmed_content.split()
    #Performing stemming on the words except stopwords
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    #Join all the twitter data(a single sentence) together
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

def sentiment_analysis():
    column_names = ["target", "id", "date", "Flag", "user", "text"]

    pickle_path = "processed_twitter_data.pkl"

    # Check if the pickle file exists
    if os.path.exists(pickle_path):
        # Load the processed DataFrame from the pickle file
        with open(pickle_path, 'rb') as file:
            twitter_data = pickle.load(file)
    else:
        #As it is reading values as 1 rows, therefore we used column name to be the first values.
        twitter_data=pd.read_csv(r"C:\Users\Lenovo\Desktop\GitHub\Chatbot\training.1600000.processed.noemoticon.csv", names=column_names ,encoding="ISO-8859-1")
        #converting values 4(positive) to 1
        twitter_data.replace({'target': {4: 1}}, inplace=True)
        #Applying stemming by calling stemming class on the data(training data)
        twitter_data['stemmed_content'] = twitter_data['text'].apply(stemming)

        # Save the processed DataFrame to a pickle file
        with open(pickle_path, 'wb') as file:
            pickle.dump(twitter_data, file)
    X = twitter_data['stemmed_content'].values
    Y = twitter_data['target'].values
    """Spliting data into test and training data, test_size => 20% test and 80% training, stratify => All the values different
    will be splitting in proper way. Means here 0 and 1 will be splitted in equal in tain and in test.
    randon_state => will split value every time in same way """
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
    global vectorizer
    vectorizer = TfidfVectorizer()
    X_train_vector_path = "X_train_vector.pkl"
    X_test_vector_path = "X_test_vector.pkl"
    vectorizer_path = "vectorizer.pkl"
    if os.path.exists(X_train_vector_path) and os.path.exists(X_test_vector_path) and os.path.exists(vectorizer_path):
        # Load the vectorized data from pickle files if they exist
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)
        with open(X_train_vector_path, 'rb') as file:
            X_train = pickle.load(file)
        with open(X_test_vector_path, 'rb') as file:
            X_test = pickle.load(file)
    else:
        # Vectorize the text data

        vectorizer = TfidfVectorizer()
        """We are converting textual data into vector numeric data, first we have to convert train data to fit_transform .
        Similarly it will convert test data accordingly similar to train data."""
        X_train = vectorizer.fit_transform(X_train)
        X_test = vectorizer.transform(X_test)

        with open(vectorizer_path, 'wb') as file:
            pickle.dump(vectorizer, file)
        # Save the vectorized data to pickle files
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
        #Training Machine learning Model
        model2 = LogisticRegression(max_iter=1000)
        model2.fit(X_train, Y_train)
        with open(Model_path, 'wb') as file:
            pickle.dump(model, file)
    data="All is good"

def sentiment_analysis_users(User_chat):
    str1=stemming(User_chat)
    X_test1 = vectorizer.transform([str1])
    prediction = model2.predict(X_test1)


    if prediction[0] == 1:
        data = "positive"
    else:
        data = "negative"
    return data

