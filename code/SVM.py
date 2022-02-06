import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import numpy as np
import sys

def main(argv=None):
        
    if argv is None:
        argv = sys.argv
     
    # Read tsv file using pandas and turn it into a dataframe
    train_df = pd.read_csv('./../results/train_features.tsv', sep="\t")
    
    # Read in dev / test set depeding on argument provided when running the python file
    if sys.argv[1] == "dev":
        test_df = pd.read_csv('./../results/dev_features.tsv', sep="\t") 
    elif sys.argv[1] == "test":
        test_df = pd.read_csv('./../results/test_features.tsv', sep="\t")

    # dropping irrelevant columns
    train_df = train_df.drop(columns=["story", "sent_index", "token_index"], axis=1)
    test_df = test_df.drop(columns=["story", "sent_index", "token_index"], axis=1)
    
    train_instances = train_df[["token", "token-2", "token-1", "token+1", "token+2", "pos", "chunk", "lemma", "matchesNeg", "hasPrefix", "hasSuffix", "hasPrefixAntonym", "hasSuffixAntonym", "matchesMulticue"]].to_dict('records')
    test_instances = test_df[["token", "token-2", "token-1", "token+1", "token+2", "pos", "chunk", "lemma", "matchesNeg", "hasPrefix", "hasSuffix", "hasPrefixAntonym", "hasSuffixAntonym", "matchesMulticue"]].to_dict('records')
    
    vec = DictVectorizer()
    X_train = vec.fit_transform(train_instances)
    
    Y_train = train_df.bio.tolist()
    Y_test = test_df.bio.tolist()
    
    classifier = LinearSVC(max_iter = 10000)
    
    parameters = dict(C=(0.01, 0.1, 1.0), loss=('hinge', 'squared_hinge'), tol=(0.0001,0.001,0.01,0.1))
    
    ### The GridSearchCV was inspired by the sklearn documentation and lecture by Ilia Markov 
    grid = GridSearchCV(estimator=classifier, param_grid=parameters, cv=5, scoring='f1_macro')
    grid.fit(X_train, Y_train)
    # Select best hyper parameters
    classifier = grid.best_estimator_
    
    X_test = vec.transform(test_instances)
    predictions = classifier.predict(X_test)
    
    test_df['predictions'] = predictions
    report = pd.DataFrame(classification_report(y_true=test_df['bio'], y_pred=test_df['predictions'], output_dict=True)).transpose()
    
    print(report)

if __name__ == '__main__':
    main()
