import nltk
import sys
from nltk import pos_tag
import pandas as pd
import spacy
from utils_A import *
from nltk.corpus import stopwords, wordnet, words, treebank
from spacy.lang.en.stop_words import STOP_WORDS

nlp = spacy.load("en_core_web_sm", disable=["tokenizer", "parser","ner", "textcat", "custom"])

multiword_expressions = [["by", "no", "means"], ["on", "the", "contrary"], ["rather", "than"], ["not", "for", "the", "world"], ["nothing", "at", "all"], ["no", "more"]]

negation_stopwords_set = {"n't", "n’t", "n‘t","cannot", "nobody", "neither", "nowhere", "not", "no", "without", "nevertheless", "nor", "never"}
# removing negation words from spacy stopwords set
new_stopwords_list = []
for i in STOP_WORDS:
    if i not in negation_stopwords_set:
        new_stopwords_list.append(i)

prefixes = {"un", "dis", "in", "a", "ab", "an", "non-", "im", "il", "ir", "anti"}
suffixes = {"less", "lessly", "lessness"}

def lemmatize_tokens(df):
    """Input dataframe. Creates lemmas for lemma column and uses spaCy lemmatizer."""
    return df.apply((lemma_wrapper),axis=1)

def part_of_speech(df):
    """Input dataframe. Creates PoS tags for column in df. spaCy PoS-tagger is used"""
    return df.apply((spacy_tagger_wrapper), axis=1)

def matches_neg_exp(negation_stopwords_set,df):
    """Input dataframe. Creates booleans for MatchesNeg-column in the dataframe. Checks for lexical negations"""
    return df['token'].apply(lambda x: x.lower() in negation_stopwords_set)
    
def element_has_prefix(token):
    """Checks whether a token contains a negation prefix. Inputs token, returns boolean."""
    prefixes_list = []
    token = token.lower()
    if check_prefix(token):
        if check_negation_prefix(token):
            return True
        else:
            return False
    else:
        return False

def has_prefix(df):
    """Input dataframe. Creates booleans for hasPrefix-column and checks for prefixal negations"""
    return df['token'].apply(element_has_prefix)

def element_has_suffix(token):
    """Checks whether a token contains a negation suffix. Inputs token, returns boolean."""
    suffixes_list = []
    token = token.lower()
    if check_suffix(token):
        if check_negation_suffix(token):
            return True
        else:
            return False
    else:
        return False
    

def has_suffix(suffixes):
    """Input dataframe. Creates booleans for hasSuffix-column and checks for suffixal negations"""
    return df['token'].apply(element_has_suffix)

def has_prefix_and_antonym(df):
    """Input dataframe. Creates booleans for hasPrefix&Antonym-column and checks for prefixal negations using WordNet antonyms"""
    return df.apply((check_antonym_prefix_wrapper), axis=1)

def has_suffix_and_antonym(df):
    """Input dataframe. Creates booleans for hasSuffix&Antonym-column and checks for suffixal negations using WordNet antonyms"""
    return df.apply((check_antonym_suffix_wrapper), axis=1)

def get_multiword(mwe_list, df):
    """Input dataframe and multiword expressions list.
    Creates matchesMulticue column in dataframe containing boolean values.
    Sets all multicue tokens to True.""" 
    # setting every value to False
    df['matchesMulticue'] = [False for x in range(len(df))]
    
    for i in range(len(df)):
        for exp in mwe_list:
            is_match = True
            # ensure index doesn't go outside dataframe
            if i + len(exp) >= len(df):
                return
            # check if token matches first word in exp or not
            if df.iloc[i]['token'] != exp[0]:
                # if not, continue to next token
                continue
            # otherwise, check if full exp matches
            for j in range(len(exp)):
                if exp[j] != df.iloc[i+j]['token']:
                    is_match = False
                    break
            # if it's a match, then change to True in dataframe
            if is_match:
                for j in range(len(exp)):
                    df.at[i + j, 'matchesMulticue'] =  True
                i = (i + len(exp)) - 1
                continue

def chunking(df):
    """Identifies the chunk that each token is a part of, based on the PoS Treebank tags.
    Input is dataframe and output is a list of chunk labels."""
    # Based on NLTK's RegexpParser tutorial: https://www.nltk.org/book_1ed/ch07.html
    grammar = r"""
    NP: {<WP|WDT|WRB>?<DT|PRP.*>?<JJ.*>?<NN.*|PRP.*>+} # noun phrase
    PP: {<IN><NP>}                                     # prepositional phrase
    VP: {<MD>?<VB.*>?<VB.*><RB.*>?<NP|PP|CLAUSE>+}    # verb phrase
    CLAUSE: {<NP><VP>}                                 # clause
    ADJP: {<RB.*>?<RB.*>?<JJ.*>}                       # adjectival phrase
    ADVP: {<RB.*><VB.*|JJ.*|NN.*>}                     # adverbial phrase
    """
    ###
    tokens = df["token"].tolist()
    # retagging as the chunker requires tuples
    pos = pos_tag(tokens)
    chunkParser = nltk.RegexpParser(grammar)
    tree = chunkParser.parse(pos)
    labels = getNodes(tree)
    return labels

def main(argv=None):
    """Creates dataframe containing all features.
    This dataframe is written to a .csv file."""
    
    if argv is None:
        argv = sys.argv
    
    argv = ['','./../data/SEM-2012-SharedTask-CD-SCO-training-simple.v2.txt']
    trainingfile = argv[1]
    
    df = pd.read_csv(trainingfile, sep="\t", names=["story", "sent_index", "token_index", "token", "bio"])
    
    # some of the features are currently quite similar,
    # we will make a selection when building the classifier
    
    df['token-2'] = df['token'].shift(2)
    df['token-1'] = df['token'].shift(1)
    df['token+1'] = df['token'].shift(-1)
    df['token+2'] = df['token'].shift(-2)
    
    df['pos'] = part_of_speech(df)
    df['hasPrefixAntonym'] = has_prefix_and_antonym(df)
    df['hasSuffixAntonym'] = has_suffix_and_antonym(df)
    df['lemma'] = lemmatize_tokens(df)
    df['chunk'] = chunking(df)
    
    df['matchesNeg'] = matches_neg_exp(negation_stopwords_set, df)
    df['hasPrefix'] = has_prefix(df)
    df['hasSuffix'] = has_suffix(df)
   
    get_multiword(multiword_expressions, df)
    
    #specifying the order of the dataframe
    new_df = df[['story', 'sent_index', 'token_index', 'token-2', 'token-1', 'token', 'token+1', 'token+2', 'pos', 'chunk', 'lemma', 'matchesNeg', 'hasPrefix', 'hasSuffix', 'hasPrefixAntonym', 'hasSuffixAntonym', 'matchesMulticue', 'bio']]
    
    # filling in NaN values
    new_new_df = new_df.fillna("X")
    
    tsvfile = 'training_features_B.tsv'
    new_new_df.to_csv(tsvfile, sep='\t')


if __name__ == '__main__':
    main()
