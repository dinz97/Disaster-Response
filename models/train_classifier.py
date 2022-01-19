# Import basic libraries
import sys
import re
import numpy as np
import pandas as pd
import pickle
from sqlalchemy import create_engine

# Import NLTK libraries
import nltk
nltk.download(['punkt', 'wordnet'])
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Import sklearn libraries
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV

def load_data(database_filepath):
    """
    INPUT:
        database_filepath - File path to database
    OUTPUT:
        X - Disaster message data
        y - Response categories of disaster messages which consist of 36 binary columns
        category_names - list of response columns' names
    """
    engine = create_engine('sqlite:///' + database_filepath)
    database_filename = database_filepath.split('/')[1].replace('.db', '')
    df = pd.read_sql('SELECT * FROM ' + database_filename, engine)
    X = df['message']
    y = df.iloc[:, 4:] # Response variables are from column index 4 till end
    category_names = y.columns.values
    
    return X, y, category_names
    


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


def build_model():
    """
    OUTPUT:
        Machine learning pipeline using Random Forest as classifier
    """
    
    # ML pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # Specify parameters for grid search
    parameters = {
        'clf__estimator__n_estimators': [50,100]
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1)
    
    return cv


def evaluate_model(model, X_test, y_test, category_names):
    """
    INPUT:
        model - Trained model
        X_test - Test disaster messages dataset
        y_test - Test disaster messages' categories
        category_names - name list of messages' categories columns
    OUTPUT:
        Model's accuracy, precision, recall, f1-score & support
    """
    
    # Predict on test data using trained model
    y_pred = model.predict(X_test)
    
    # Model performance
    accuracy = round(((y_pred == y_test).mean().mean()) * 100, 1)
    print('Accuracy (%): ', accuracy )
    print(classification_report(y_test.values, y_pred, target_names=category_names))


def save_model(model, model_filepath):
    """
    INPUT:
        model - Trained model
        model_filepath - File path to model
    OUTPUT:
        Model in .pkl format
    """
    
    # Save model
    pickle.dump(model, open(model_filepath, 'wb'))
    


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()