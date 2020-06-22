from flask import Flask,render_template,request,redirect


import numpy as np
import joblib

model = joblib.load('model2.pkl')

cv = joblib.load('vector2.pkl')

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer

ps = PorterStemmer()
stemmer = PorterStemmer()
tokenizer = RegexpTokenizer('[a-z0-9]+')
sw = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

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
