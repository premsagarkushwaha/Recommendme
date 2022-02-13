from flask import Flask,render_template,request

import nltk
nltk.download('stopwords')
import pandas as pd
df = pd.read_csv('tmdb_5000_movies.csv')
df = df[['title', 'tagline', 'overview', 'popularity']]
df["title"] = df["title"].str.lower()
df.tagline.fillna('', inplace=True)
df['description'] = df['tagline'].map(str) + ' ' + df['overview']
df.dropna(inplace=True)
df = df.sort_values(by=['popularity'], ascending=False)

import re
import numpy as np
import contractions

stop_words = nltk.corpus.stopwords.words('english')
def normalize_document(doc):
    doc = re.sub(r'[^a-zA-Z0-9\s]', '', doc, re.I|re.A)
    doc = doc.lower()
    doc = doc.strip()
    doc = contractions.fix(doc)
    tokens = nltk.word_tokenize(doc)
    filtered_tokens = [token for token in tokens if token not in stop_words]
    doc = ' '.join(filtered_tokens)
    return doc
normalize_corpus = np.vectorize(normalize_document)
norm_corpus = normalize_corpus(list(df['description']))
from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
tfidf_matrix = tf.fit_transform(norm_corpus)
tfidf_matrix.shape
from sklearn.metrics.pairwise import cosine_similarity
doc_sim = cosine_similarity(tfidf_matrix)
doc_sim_df = pd.DataFrame(doc_sim)
movies_list = df['title'].values
def movie_recommender(movie_title, movies=movies_list, doc_sims=doc_sim_df):
    movie_idx = np.where(movies == movie_title)[0][0]
    movie_similarities = doc_sims.iloc[movie_idx].values
    similar_movie_idxs = np.argsort(-movie_similarities)[1:6]
    similar_movies = movies[similar_movie_idxs]
    return similar_movies



app = Flask(__name__)

@app.route("/")
@app.route("/home")
def home():
    return render_template('index2.html')   

@app.route("/result",methods = ['POST',"GET"])
def result():
    
    output = request.form.to_dict()
    name = output["name"]
    if name not in movies_list:
        return render_template('index.html')
    else:

        nl = movie_recommender(movie_title=name)

        lists = list(nl)
        return render_template('index2.htmlac',name = lists)

if __name__ == "__main__":
    app.run(debug=True,port=8000)