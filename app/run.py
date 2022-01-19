import re
import json
import plotly
import pandas as pd
import joblib

import nltk
nltk.download(['punkt', 'wordnet'])
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    """
    INPUT:
        text - Text message from Disaster message data 
    OUTPUT:
        cleansed_tokens  - Tokenised text message from Disaster message data
    """
    
    # Remove punctutation characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    
    # Split text into words using NLTK
    tokens = word_tokenize(text)
    
    # Remove stop words
    tokens = [w for w in tokens if w not in stopwords.words("english")]
    
    # Lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Reduce words to their root form
    cleansed_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]
    
    # Lemmatize verbs by specifiying pos
    cleansed_tokens = [lemmatizer.lemmatize(tok, pos='v').lower().strip() for tok in cleansed_tokens]
    
    return cleansed_tokens

# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')

def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    # Table for Distribution of Message Genres
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # Table for Distribution of Messages Categories
    categories_counts = df.iloc[:, 4:].sum()
    categories_names = df.iloc[:, 4:].columns
    
    # Table for Top 10 Message categories
    top10_cat_counts = df.iloc[:,4:].mean().sort_values(ascending=False)[0:10]
    top10_cat_names = list(top10_cat_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        # Graph 1 - Distribution of Messages Genres
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        
        # Graph 2 - Distribution of Messages Categories
        {
            'data': [
                Bar(
                    x=categories_names,
                    y=categories_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        },
        
        # Graph 3 - Top 10 Messages Categories
        {
            'data': [
                Bar(
                    x=top10_cat_names,
                    y=top10_cat_counts
                )
            ],

            'layout': {
                'title': 'Top 10 Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        },
        
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()