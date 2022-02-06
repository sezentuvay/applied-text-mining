### THE CODE FOR THIS ALGORITHM WAS TAKEN FROM HERE: https://github.com/cltl/ba-text-mining/blob/master/lab_sessions/lab4/Lab4a.4-NERC-CRF-Dutch.ipynb
### AS WELL AS THE SKLEARN DOCUMENTATION, AND SUBSEQUENTLY ADJUSTED FOR OUR SPECIFIC DATA!

# Copyright: Vrije Universiteit Amsterdam, Faculty of Humanities, CLTL

### NOTE: YOU NEED TO HAVE SKLEARN-VERSION < 0.24 IN ORDER TO RUN PART OF THE CODE
### LATER VERSIONS WILL NOT WORK, SEE: https://github.com/TeamHG-Memex/sklearn-crfsuite/issues/60

import sklearn
import csv
import sys
import pandas as pd
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats
from sklearn.metrics import make_scorer
import sys

def extract_sents_from_tsv(inputfile):
    """Reads in datafile and returns sentences 
    with features (as tuples in a list).
    
    Input: filepath to .tsv file
    Output: list of sentences"""
    sents = []
    current_sent = []

    with open(inputfile, "r") as infile:
        reader = csv.reader(infile)
        next(reader)
        for line in infile:
            # using tsv files here as the csv files get split incorrectly 
            row = line.strip("\n").split('\t')
            if row[4] == ".":
                current_sent.append(tuple(row))
                sents.append(current_sent)
                current_sent = []
            else:
                current_sent.append(tuple(row))
    return sents

def token2features(sentence, i):
    
    
    # comment out features you do not want to include
    story = sentence[i][1]
    sent_index = sentence[i][2]
    token_index = sentence[i][3]
    prev_prev_token= sentence[i][4]
    prev_token = sentence[i][5]
    token = sentence[i][6]
    next_token = sentence[i][7]
    next_next_token = sentence[i][8]
    pos = sentence[i][9]
    chunk = sentence[i][10]
    lemma= sentence[i][11]
    matchesNeg = sentence[i][12]
    hasPrefix = sentence[i][13]
    hasSuffix = sentence[i][14]
    hasPrefixAntonym = sentence[i][15]
    hasSuffixAntonym = sentence[i][16]
    matchesMulticue = sentence[i][17]

    # comment out features you do not want to include
    features = {
        'bias': 1.0,
        'story':story,
        'sent_index':sent_index,
        'token_index':token_index,
        'token-2':prev_prev_token,
        'token-1':prev_token,
        'token': token,
        'token+1':next_token,
        'token+2': next_next_token,
        'pos': pos,
        'chunk':chunk,
        'lemma':lemma,
        'matchesNeg':matchesNeg,
        'hasPrefix':hasPrefix,
        'hasSuffix':hasSuffix,
        'hasPrefixAntonym':hasPrefixAntonym,
        'hasSuffixAntonym':hasSuffixAntonym,
        'matchesMulticue':matchesMulticue
    }
        
    return features

def sent2features(sent):
    """
    Extracts features for each sentence.
    Input: sentence list
    Output: nested list with feature dictionaries for each token in sentence
    """
    return [token2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    """
    Extracts gold labels for each sentence.
    Input: sentence list
    Output: list with labels list for each token in the sentence
    """
    # gold labels at index 18
    return [word[18] for word in sent]

def main(argv=None):
        
    if argv is None:
        argv = sys.argv
        
    train_sents = extract_sents_from_tsv("./../results/train_features.tsv")
    if sys.argv[1] == "dev":
        test_sents = extract_sents_from_tsv("./../results/dev_features.tsv")
    elif sys.argv[1] == "test":
        test_sents = extract_sents_from_tsv("./../results/test_features.tsv")
        
    # crf works with input and output sequences
    X_train = [sent2features(s) for s in train_sents]
    Y_train = [sent2labels(s) for s in train_sents]

    X_test = [sent2features(s) for s in test_sents]
    Y_test = [sent2labels(s) for s in test_sents]
    
    ### Taken from the sklearn documentation: https://sklearn-crfsuite.readthedocs.io/en/latest/tutorial.html#features
    crf = sklearn_crfsuite.CRF(algorithm='lbfgs', max_iterations=100, all_possible_transitions=True)
    params_space = {'c1': scipy.stats.expon(scale=0.5), 'c2': scipy.stats.expon(scale=0.05)}

    # use the same metric for evaluation
    f1_scorer = make_scorer(metrics.flat_f1_score, average='weighted')

    # search
    rs = RandomizedSearchCV(crf, params_space, cv=3, verbose=1, n_jobs=-1, n_iter=50, scoring=f1_scorer, random_state=0)
    rs.fit(X_train, Y_train)
    
    # select best hyperparameters    
    crf = rs.best_estimator_
    
    labels = list(crf.classes_)
    Y_pred = crf.predict(X_test)
    
    sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))
    
    #dataframe to create classification report for individual tokens
    results_df = pd.DataFrame()
    
    # adding predicted label for each individual token in dataframe series
    predicted_labels = [label for small_list in Y_pred for label in small_list]
    results_df['pred_labels'] = predicted_labels
    
    # adding gold label for each individual token in dataframe series
    gold_labels = [label for small_list in Y_test for label in small_list]
    results_df['gold_labels'] = gold_labels
    
    # also adding tokens so it's easy to analyze the results per token
    test_tokens = [token_tuple[6] for small_list in test_sents for token_tuple in small_list]
    results_df['tokens'] = test_tokens
    
    report = pd.DataFrame(classification_report(y_true=results_df['gold_labels'], y_pred=results_df['pred_labels'], labels=sorted_labels, output_dict=True)).transpose()   
    
    print(report)


if __name__ == '__main__':
    main()
