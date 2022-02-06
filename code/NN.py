# Necessary imports
import gensim
import spacy
import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV # For optimization
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# Change this to your own path of a word embedding model
google_news_path = "../Downloads/GoogleNews-vectors-negative300.bin.gz"

# Load google news embeddings using gensim
word_embedding_model = gensim.models.KeyedVectors.load_word2vec_format(google_news_path, binary=True)

# Set this to the amount of dimensions in the used word embedding model
num_features = 300

def find_embed(word):
    '''
    Returns the word embedding for a word if it exists, otherwise returns a list full of zeros
    '''
    try:
        return(word_embedding_model[word.lower()])
    except:
        return [0]*num_features
    
def main(argv=None):
        
    if argv is None:
        argv = sys.argv
    
    trainingfile = './../data/SEM-2012-SharedTask-CD-SCO-training-simple.v2.txt'
    
    # Use pandas to read in (txt) file into pandas dataframe
    df_train = pd.read_csv(trainingfile, sep="\t", names=["story", "sent_index", "token_index", "token", "bio"])
    
    # Find corresponding word embeddings for every token
    df_train['embedding'] = df_train['token'].apply(find_embed)
    
    
    X = np.array(df_train['embedding'].tolist())
    y = np.array(df_train['bio'])

    #scikit-learn's train_test_split() to test the system on the training data (75/25 division)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    
    # Use GridSearch to find the best parameters
    mlp = MLPClassifier(max_iter=10000)
    parameter_space = {
        'solver': ['lbfgs', 'sgd', 'adam'],
        'alpha': [0.0001, 0.05, 0.01],
    }

    clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
    clf.fit(X_train, y_train)
    
    best_alpha = clf.best_params_['alpha']
    best_solver = clf.best_params_['solver']

    mlp = MLPClassifier(max_iter=10000,alpha = best_alpha, solver = best_solver)
    mlp.fit(X_train, y_train)
    predictions = mlp.predict(X_test)

    print(classification_report(y_test, predictions, digits=5))
    
if __name__ == '__main__':
    main()
