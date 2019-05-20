from flask import Flask, render_template, request, redirect, url_for, session
from flask_session import Session
from form import MyForm

from flask import jsonify

from logging import FileHandler, WARNING

#import text_analysis
import preprocessing

import PyPDF2
import os

import string
import re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# LIBRARY FOR RANKED RETRIEVAL
import math
from collections import OrderedDict

import preprocessing

app = Flask(__name__)

file_handler = FileHandler('errorlog.txt') #file_handler logs errors
file_handler.setLevel(WARNING)
app.config.from_object(__name__) # Config for Flask-Session

app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024 #max upload size is 1MB
app.config['SECRET_KEY'] = 'secret'
app.config['ALLOWED_EXTENSIONS'] = set(['txt'])#restricts file extensions to the .txt extension
app.config['SESSION_TYPE'] = 'filesystem' #config for Flask Session, indicates will store session data in a filesystem folder
app.logger.addHandler(file_handler)
Session(app) #create Session instance

def allowed_file(filename):
    '''Checks uploaded file to make sure it is.txt'''
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=('GET', 'POST'))
def index():
    form = MyForm()
    if form.validate_on_submit():
        session['file_contents'] = form.name.data
        return redirect('/analysis')
    return render_template('submit.html', form=form)

@app.route('/submit', methods=('GET', 'POST'))
def submit():
    form = MyForm()
    if form.validate_on_submit():
        session['file_contents'] = form.name.data
        return redirect('/analysis')
    return render_template('submit.html', form=form)

@app.route('/load')
def load():
    return render_template('submit.html')

# @app.route('/upload', methods = ['GET', 'POST'])
# def upload():
#     if request.method == 'POST':
#         if 'file' not in request.files:
#             return redirect(request.url)
#         file = request.files['file']
#         if file.filename == '':
#             return redirect(request.url)
#         if file and allowed_file(file.filename):
#             session['file_contents'] = file.read().decode('utf-8')
#             return redirect(url_for('analysis'))
#     return render_template('upload.html')

# @app.route('/analysis')
# def analysis():
#     file_contents = session.get('file_contents')
#     analyzed_sent = text_analysis.get_pos_neg(file_contents)
#     pos_sent = analyzed_sent[-1]
#     neg_sent = analyzed_sent[0]
#     word_cloud = text_analysis.render_word_cloud(file_contents)
#     return render_template('analysis.html', pos_sent=pos_sent, neg_sent=neg_sent, word_cloud=word_cloud)

@app.route('/loadData')
def loadData():
    location = 'biografi'
    filename = preprocessing.allFile(location) #jalan
    extracted= preprocessing.extractPDF(location) #jalan
    totalDoc = len(filename)
    documentNumber = preprocessing.generateDocNumber(filename)

    for i in range(len(filename)):
         extracted[i] = str(extracted[i].encode("utf-8"))
    
    # # PREPROCESSING
    text = preprocessing.removePunctuation(extracted)
    text = preprocessing.caseFolding(text)
    text = preprocessing.tokenize(text)
    text = preprocessing.stopwordRemove(text)
    text = preprocessing.numberRemove(text)
    text = preprocessing.stemming(text)

    # # GET ALL TERMS IN COLLECTION
    terms = preprocessing.getAllTerms(text)

    # # INDEXING

    # # index = createIndex(text,documentNumber, terms)
    index = preprocessing.createIndex(text,documentNumber)

    query_search = session.get('file_contents')
    
    query = preprocessing.removePunctuation(query_search)
    query = preprocessing.caseFolding(query)
    query = preprocessing.tokenize(query)
    query = preprocessing.stopwordRemove(query)
    query = preprocessing.numberRemove(query)
    query = preprocessing.stemming(query)
    #query = query[0]

    # # Check Query In Index
    query = preprocessing.queryInIndex(query, index)

    return ("Index Berhasil")

@app.route('/analysis')
def analysis():
    location = 'biografi'
    filename = preprocessing.allFile(location) #jalan
    extracted= preprocessing.extractPDF(location) #jalan
    totalDoc = len(filename)
    documentNumber = preprocessing.generateDocNumber(filename)

    for i in range(len(filename)):
        extracted[i] = str(extracted[i].encode("utf-8"))
    

    # # # PREPROCESSING
    text = preprocessing.removePunctuation(extracted)
    
    text = preprocessing.caseFolding(text)
    text = preprocessing.tokenize(text)
    text = preprocessing.stopwordRemove(text)
    text = preprocessing.numberRemove(text)
    text = preprocessing.stemming(text)

    # # # GET ALL TERMS IN COLLECTION
    terms = preprocessing.getAllTerms(text)


    # # # INDEXING

    # # # index = createIndex(text,documentNumber, terms)
    index = preprocessing.createIndex(text, documentNumber)
    
    query_search = session.get('file_contents')
    
    query = preprocessing.removePunct(query_search)
    
    query = preprocessing.caseFold(query)
    
    query = preprocessing.tokenization(query)
    
    query = preprocessing.stopwordRemove(query)
    query = preprocessing.numberRemove(query)
    query = preprocessing.stemming(query)
    query = query[0]
    
    # # # # Check Query In Index
    query = preprocessing.queryInIndex(query, index)
    # return ("index query berhasil")
    # # #return str(text)
    # return ("Index berhasil")
    

    # RANKED RETRIEVAL

    
    N               = totalDoc
    tfidf_list      = []

    docFrequency    = preprocessing.df(query, index)
    invDocFrequency = preprocessing.idf(docFrequency, N)
    termFrequency   = preprocessing.tf(query, index)
    TFIDF           = preprocessing.tfidf(termFrequency, invDocFrequency)
    sc              = preprocessing.score(TFIDF)

    print(len(sc))
    relevanceDocNumber = []
    count = 0

    print('Query: ', query_search,'\n\n')
    print('Result: \n')
    # for i in range(5):
    #     a = documentNumber.index(sc[i])
    #     print('Document Number: ',sc[i])
    #     print(filename[a])
    #     print('-------------------------------------------\n')

    result = []
    for i in range(len(sc)):
        relevanceDocNumber.append(int(sc[i]))
        a = documentNumber.index(sc[i])
        print()
        print('==========================================================================\n')
        print('| Filename: ',filename[a],' | Document ID: ',documentNumber[a],'|','\n')
        print(extracted[a][0:1000])
        print('\n==========================================================================')
        print('\n\n\n')
        # count = count + 1
        result.append((filename[a], documentNumber[a]))
        if(i >= 5):
            break

    return render_template('load.html', a=documentNumber[a], qs=query_search, fl=filename[a], hasil = result)


# @app.route('/about')
# def about():
#     return render_template('about.html')

if __name__ == '__main__':
   app.run()
