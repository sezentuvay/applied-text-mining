import sys
import pandas as pd
from utils_B import *
import nltk
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet, words, treebank
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()

stopwords_list = stopwords.words('english')

multiword_expressions = [["by", "no", "means"], ["on", "the", "contrary"], ["rather", "than"], ["not", "for", "the", "world"], ["nothing", "at", "all"], ["no", "more"]]

negation_stopwords_set = ["nobody", "neither", "nowhere", "not", "no", "without", "nevertheless", "nor", "never", "ain", "aren", "aren't", 'couldn', "couldn't", "didn", "didn't", "doesn", "doesn't", \
"hadn", "hadn't", "hasn", "hasn't", "haven't","isn", "isn't", "mightn", "mightn't", "mustn", "mustn't", "needn", "needn't", "shan", "shan't", "shouldn", "shouldn't", "wasn", "wasn't",\
 "weren", "weren't", "won't", "wouldn", "wouldn't", "don't", "'t"]

# filtering out negations that are in the NLTK stopwords list
new_stopwords_list = []
for i in stopwords_list:
    if i not in negation_stopwords_set:
        new_stopwords_list.append(i)

prefixes = {"un", "dis", "in", "a", "ab", "an", "non-", "im", "il", "ir", "anti"}
suffixes = {"less", "lessly", "lessness"}

def lemmatize_tokens(df):
    """Input dataframe. Creates lemmas for lemma column and uses NLTK lemmatizer."""
    return df.apply((lemma_wrapper),axis=1)

def part_of_speech(df):
    """Input dataframe. Creates PoS tags for column in df. NLTK PoS-tagger is used"""
    token_list = df['token'].tolist()
    pos_token_list = pos_tag(token_list)
    pos_list = [tuple[1] for tuple in pos_token_list]
    return pos_list

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
    
def has_suffix(df):
    """Inputs dataframe. Creates booleans for hasSuffix-column and checks for suffixal negations"""
    return df['token'].apply(element_has_suffix)

def has_prefix_and_antonym(df):
    """Inputs dataframe. Creates booleans for hasPrefix&Antonym-column and checks for prefixal negations using WordNet antonyms"""
    return df.apply((check_antonym_prefix_wrapper), axis=1)

def has_suffix_and_antonym(df):
    """Inputs dataframe. Creates booleans for hasSuffix&Antonym-column and checks for suffixal negations using Wordnet antonyms"""
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
        
    trainingfile = './../data/SEM-2012-SharedTask-CD-SCO-training-simple.v2.txt'
    devfile = './../data/SEM-2012-SharedTask-CD-SCO-dev-simple.v2.txt'
    testfile_1 = './../data/SEM-2012-SharedTask-CD-SCO-test-cardboard.txt'
    testfile_2 = './../data/SEM-2012-SharedTask-CD-SCO-test-circle.txt'
    
    if sys.argv[1] == "train":
        df = pd.read_csv(trainingfile, sep="\t", names=["story", "sent_index", "token_index", "token", "bio"])
    elif sys.argv[1] == "dev":
        df = pd.read_csv(devfile, sep="\t", names=["story", "sent_index", "token_index", "token", "bio"])
    elif sys.argv[1] == "test":
        # combining the two test sets into one
        df_1 = pd.read_csv(testfile_1, sep="\t", names=["story", "sent_index", "token_index", "token", "bio"])
        df_2 = pd.read_csv(testfile_2, sep="\t", names=["story", "sent_index", "token_index", "token", "bio"]) 
        df = df_1.append(df_2, ignore_index=True)    
    
    # 'shift' function shifts the index, puts NaN values at empty indices
    df['token-2'] = df['token'].shift(2)
    df['token-1'] = df['token'].shift(1)
    df['token+1'] = df['token'].shift(-1)
    df['token+2'] = df['token'].shift(-2)
    df['pos'] = part_of_speech(df)
    
    # need to convert to WordNet PoS tags for the following functions
    df["pos"] = df["pos"].apply(get_wordnet_pos)
    df['hasPrefixAntonym'] = has_prefix_and_antonym(df)
    df['hasSuffixAntonym'] = has_suffix_and_antonym(df)
    df['lemma'] = lemmatize_tokens(df)
    
    # changing column back to original Treebank PoS tags
    df['pos'] = part_of_speech(df)
    df['chunk'] = chunking(df)
    
    df['matchesNeg'] = matches_neg_exp(negation_stopwords_set, df)
    df['hasPrefix'] = has_prefix(df)
    df['hasSuffix'] = has_suffix(df)
   
    get_multiword(multiword_expressions, df)
    
    #specifying the order of the dataframe
    new_df = df[['story', 'sent_index', 'token_index', 'token-2', 'token-1', 'token', 'token+1', 'token+2', 'pos', 'chunk', 'lemma', 'matchesNeg', 'hasPrefix', 'hasSuffix', 'hasPrefixAntonym', 'hasSuffixAntonym', 'matchesMulticue', 'bio']]
    
    # filling in NaN values created by use of 'shift' function
    new_new_df = new_df.fillna("X")
    
    if sys.argv[1] == "train":
        tsvfile = './../results/train_features.tsv'
    elif sys.argv[1] == "dev":
        tsvfile = './../results/dev_features.tsv'
    elif sys.argv[1] == "test":
        tsvfile = './../results/test_features.tsv'
    
    new_new_df.to_csv(tsvfile, sep='\t')


if __name__ == '__main__':
    main()
