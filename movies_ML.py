from array import array
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.stem.porter import PorterStemmer

# Load the dataset
df = pd.read_csv('movie_dataset.csv',delimiter=',')

df = df.dropna() # drop rows with nans

df.columns 

# Create the corpus we are working with and used as predictor
df['corpus'] = df.genres + ' ' + df.keywords + ' ' + df.cast + ' ' + df.director

# Create a new dataframe with the x and y variables
df = df[['corpus','title']].copy()

# Clean the corpus we are working with
 
porter = PorterStemmer()
stop = nltk.corpus.stopwords.words("english")

def clean_stop(string):
    regex2=r"\d|[^\w\s]" # Deletes signs and numbers 
    string = re.sub(regex2, '', string) 
    return string

stop = [clean_stop(i) for i in stop]

def tokenizer(text):
    return nltk.word_tokenize(text,"english")

def tokenizer_porter(text):
    return [porter.stem(word) for word in tokenizer(text)]

def preprocessor2(text):
    return " ".join([w for w in tokenizer_porter(text) if w not in stop])

def clean_string(string):
    regex = r"\(.*?\)"  # Deletes text in parenthesis
    string = re.sub(regex, '', string) 
    regex2=r"\d|[^\w\s]" # Deletes signs and numbers 
    string = re.sub(regex2, '', string) 
    string = " ".join(string.split()) # Deletes double spaces
    string = string.lower()
    string  = preprocessor2(string)
    return string

df['corpus'] = df['corpus'].apply(clean_string)

# Split the data to train the model

X_train, X_test, y_train, y_test = train_test_split( df["corpus"].values, df["title"].values, test_size=0.3)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

# Train the model (Logstic Regression)

tfidf = TfidfVectorizer()
LR = LogisticRegression()
pipe = Pipeline([('vect', tfidf),('clf', LR)])  # armo un pipeline, paso la funcion tfidf (vectorizo los datos de x) y luego uso la funcion de regression.


pipe.fit(X_train,y_train)

# Input the: genres, keywords, cast and director, to feed the model and predict a movie based on those parameters


X_test[0] # testing using Interstellar movie

pipe.predict(np.array([X_test[0]]))

text = ['funny comedy laugh']

pipe.predict(np.array(text))
# IMPORTANT => check not to split the dataset, the movies do not repeat it does not make sense to split the data we are losing information