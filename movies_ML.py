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

# Clean up the corpus we are working with
 
porter = PorterStemmer()
stop = nltk.corpus.stopwords.words("english")

def clean_stop(string): # Clean a little bit the stop words
    regex2=r"\d|[^\w\s]" # Deletes signs and numbers 
    string = re.sub(regex2, '', string) 
    return string

stop = [clean_stop(i) for i in stop] # New list of stop words (cleaned)

def tokenizer(text): # Tokenize the words
    return nltk.word_tokenize(text,"english")

def tokenizer_porter(text): # Stem the tokenized words (reduce words to their root)
    return [porter.stem(word) for word in tokenizer(text)]

def preprocessor2(text): # Join back the tokenized and stemmed word to a single string
    return " ".join([w for w in tokenizer_porter(text) if w not in stop])

def clean_string(string): # Defining a single function to clean the string
    regex = r"\(.*?\)"  # Deletes text in parenthesis
    string = re.sub(regex, '', string) 
    regex2=r"\d|[^\w\s]" # Deletes signs and numbers 
    string = re.sub(regex2, '', string) 
    string = " ".join(string.split()) # Deletes double spaces
    string = string.lower() # Lowers the string
    string  = preprocessor2(string) # Applies the preprocessor to the whole string
    return string

# Apply the cleaning function to the corpus

df['corpus'] = df['corpus'].apply(clean_string)

# Set the dependent and independent variables to be used in the model (in this case we do not split the data in train and test)

X_train = df.corpus.values
y_train = df.title.values

# Train the model (Logstic Regression)

tfidf = TfidfVectorizer()
LR = LogisticRegression()
pipe = Pipeline([('vect', tfidf),('clf', LR)])  # armo un pipeline, paso la funcion tfidf (vectorizo los datos de x) y luego uso la funcion de regression.


pipe.fit(X_train,y_train)

# Input the: genres, keywords, cast and director, to feed the model and predict a movie based on those parameters

text = ['galaxy space spaceship']

pipe.predict(np.array(text))
