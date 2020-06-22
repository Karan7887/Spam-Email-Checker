from flask import Flask,render_template,request,redirect


import numpy as np
import joblib

model = joblib.load('model2.pkl')

cv = joblib.load('vector2.pkl')

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

ps = PorterStemmer()
stemmer = PorterStemmer()
tokenizer = RegexpTokenizer('[a-z0-9]+')
sw = set(stopwords.words('english'))

def transfor(message):
    message = message.lower()
    message = tokenizer.tokenize(message)
    message = [stemmer.stem(j) for j in message if j not in sw]
    message = ' '.join(message)
    check = [message]
    return check
    

app = Flask(__name__)

@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/',methods = ['POST'])
def pred():
    if request.method == 'POST':
        message = request.form['review']
        message = transfor(message)
        message = cv.transform(message)
        prediction = model.predict(message)
        prediction = prediction
        return render_template('index.html',your_pred = prediction)

if __name__ == '__main__':
	app.run(debug=True)
