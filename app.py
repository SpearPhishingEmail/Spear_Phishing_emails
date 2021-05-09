from flask import Flask
from flask import Flask, render_template, request, url_for, flash, redirect
from sqlalchemy import create_engine
import pymysql
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import operator
import string
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.data import load
from nltk.stem import SnowballStemmer
from string import punctuation
import re
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_score
from sklearn.utils import shuffle

punctuation = list(string.punctuation)
stemmer = SnowballStemmer("english")
stop = stopwords.words('english') + punctuation
########################################
emoticons_str = r"""
    (?:
        [:=;] # :(
        [oO\-]? # No se
        [D\)\]\(\]/\\OpP] # 
    )"""
 
regex_str = [
    emoticons_str,
    r'<[^>]+>', # HTML tags
    r'(?:@[\w_]+)', # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
 
    r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    r'(?:[\w_]+)', # other words
    r'(?:\S)', # anything else
    r'[0-9]+'
]

def get_dataset():
    data = pd.read_csv("dataset_fraude.csv", sep=";", encoding='windows-1252')
    shuffle(data)
    Email = data["email"]
    Label = data["label"]
    emails=np.asarray(Email)
    return emails, Label

####Quitar numeros##############
def remove(list): 
    pattern = '[0-9]'
    list = [re.sub(pattern, '', i) for i in list]
    return list
#######################################
def generate_ngrams(s, n):
    # Convert to lowercases
    s = s.lower()
    # Romper la oración en el tokens
    tokens_first = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE )
    tokens = tokens_first.findall(s)
        
    # eliminar tokens vacíos, links, menciones, stopwords y hastags
    tokens = [token for token in tokens if (token != "" or token != '' ) and (token not in stop) and (not token.startswith(('#', '@', 'https:', 'http:','<')))]
    #lematizar
    tokens = [stemmer.stem(token) for token in tokens]
    tokens = remove(tokens)
    tokens = [token for token in tokens if (token != "" or token != ''  ) and (token not in stop) and (not token.startswith(('#', '@', 'https:', 'http:')))]
#     print(tokens)
    # funcion zip genera n-grams
    # Concatena los tokens en los ngrams y los retorna
    ngrams = zip(*[tokens[i:] for i in range(n)])
    return [" ".join(ngram) for ngram in ngrams]



#########################################
#############Documentos##################
def clean_text(emails,text):
    bigBagOfWords = []
    unique = []
    ##########Preprocesamiento##########
    n=1
    for email in emails:
        bagOfWords = generate_ngrams(str(email), n)
        unique = set(unique).union(set(bagOfWords))
        bigBagOfWords.append(bagOfWords)

    bagOfWords = generate_ngrams(str(text), n)
    unique = set(unique).union(set(bagOfWords))
    bigBagOfWords.append(bagOfWords)
    return bigBagOfWords
def calculate_tfidf(bigBagOfWords,Label):
    n = [" ".join(ngram) for ngram in bigBagOfWords]
    vectorizer = TfidfVectorizer(ngram_range=(1, 1),stop_words=stop, use_idf=True)
    X = vectorizer.fit_transform(n)
    vocabulary = vectorizer.get_feature_names()
    df = pd.DataFrame(data=X.toarray(), columns=vocabulary).iloc[:,0::2]
    x = df.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)
    df['label'] = Label
    df.to_csv (r'features unigrama.csv', index = False, header=True)
    return df.tail(1)

def train_dataset():
    dataset = pd.read_csv("Features unigrama.csv", sep=",", low_memory=False)
    shuffle(dataset)
    df = dataset.fillna(0)
    #Create a RandomForest Classifier
    clf=RandomForestClassifier(n_estimators=1000,bootstrap = True,oob_score=True, n_jobs=-1)
    y = df.label
    X = df.drop('label', axis=1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state=5)
    clf.fit(X, y)
    return clf
    #print(clf.predict(text_vector))

app = Flask(__name__)
app.secret_key = b'\xed\xc3X\xc4\xd6\xe6 1\xb2=@\x08m\x11U\x1a'
app.config['TESTING'] = True


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/phishing', methods=('GET', 'POST'))
def create():
    emails, label = get_dataset()
    if request.method == 'POST':
        text = request.form['content']
        if not text :
            flash('Text is necessary')
        else:
            bigBagOfWords = clean_text(emails,text)
            text_tfidf=calculate_tfidf(bigBagOfWords,label)
            print(text_tfidf.drop('label', axis=1))
            test_vector = text_tfidf.drop('label', axis=1)
            print(test_vector.to_numpy())
            clf = train_dataset()
            print(clf.predict(test_vector.to_numpy()))
            prediction = clf.predict(test_vector.to_numpy())
            if int(prediction[0]) == 0:
                flash(f"The email '{text}' is NOT pishing")
            else:
                flash(f"The email '{text}' is pishing")

            return redirect(url_for('index'))
    return render_template('phishing.html')