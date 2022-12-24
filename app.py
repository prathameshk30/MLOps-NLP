from flask import Flask, render_template, url_for, request
import pandas as pd
import numpy as np
from nltk.stem.porter import PorterStemmer
import re
import string
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

def count_punct(text):
    count= sum([1 for char in text if char in string.punctuation])
    return round(count/len(text)-text.count(" "),3)*100


from logging import FileHandler,WARNING
app= Flask(__name__, template_folder = 'templates')

file_handler = FileHandler('errorlog.txt')
file_handler.setLevel(WARNING)

filename = 'finalized_model.sav'
clf = pickle.load(open(filename, 'rb'))

filename = 'vectorizer.sav'
cvs = pickle.load(open(filename, 'rb'))

@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predict',methods=['GET', 'POST'])
def predict():
    if request.method== 'POST':
        message= request.form['message']
        data= [message]
        vect= pd.DataFrame(cvs.transform(data).toarray())
        body_len=pd.DataFrame([len(data) - data.count(" ")])
        punct= pd.DataFrame([count_punct(data)])
        total_data= pd.concat([body_len, punct,vect], axis=1)
        my_prediction= clf.predict(total_data)
    return render_template('result.html',prediction=my_prediction)


if __name__=="__main__":
    app.debug = True
    app.run(port=4000, host='0.0.0.0')