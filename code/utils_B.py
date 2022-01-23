from nltk import pos_tag
import pandas as pd
import nltk
from nltk.corpus import stopwords, wordnet, words, treebank
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()

stopwords_list = stopwords.words('english')
negation_stopwords_set = ["nobody", "neither", "nowhere", "not", "no", "without", "nevertheless", "nor", "never", "ain", "aren", "aren't", 'couldn', "couldn't", "didn", "didn't", "doesn", "doesn't", \
"hadn", "hadn't", "hasn", "hasn't", "haven", "haven't","isn", "isn't", "mightn", "mightn't", "mustn", "mustn't", "needn", "needn't", "shan", "shan't", "shouldn", "shouldn't", "wasn", "wasn't",\
 "weren", "weren't", "won't", "wouldn", "wouldn't", "don", "don't", "'t"]
new_stopwords_list = []
for i in stopwords_list:
    if i not in negation_stopwords_set:
        new_stopwords_list.append(i)

prefixes = {"un", "dis", "in", "a", "ab", "an", "non-", "im", "il", "ir", "anti"}
suffixes = {"less", "lessly", "lessness"}

# TAKEN FROM: https://stackoverflow.com/questions/14841997/how-to-navigate-a-nltk-tree-tree
def getNodes(parent):
    """Gets chunk labels from syntactic tree"""
    labels = []
    for node in parent:
        if type(node) is nltk.Tree:
            for leave in node.leaves():
                labels.append(node.label())
        else:
            labels.append("no label")
    return labels
###

def delete_one_prefix(token, prefixes):
    """The prefixes are deleted from an input token, using the prefixes-set. Output is the remaining token"""
    for prefix in prefixes:
        token = token.lower()
        if token.startswith(prefix) and token not in new_stopwords_list:
            token = token.strip(prefix)
    return token

def delete_one_suffix(token, suffixes):
    """The suffixes are deleted from an input token, using the suffixes-set. Output is the remaining token"""
    for suffix in suffixes:
        token = token.lower()
        if token.endswith(suffix) or token not in new_stopwords_list:
            token = token.strip(suffix)
    return token

def lemma_wrapper(row):
    """Wrapper for lemmatizer function"""
    return nltk_lemmatizer_token(str(row.token), row.pos)

def nltk_lemmatizer_token(token, pos):
    """One input token is lemmatized using NLTK. The lemma is returned"""
    lemma = wnl.lemmatize(token,pos=pos)
    return lemma

### Taken from: https://stackoverflow.com/questions/15586721/wordnet-lemmatization-and-pos-tagging-in-python
def get_wordnet_pos(treebank_tag):
    """Converts Treebank to Wordnet PoS-tag. Helps accuracy of WordNetLemmatizer."""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
###

def find_antonyms_wrapper(row):
    """Wrapper for find_antonyms function"""
    return find_antonyms(str(row.token), str(row.pos))

def check_antonym_prefix_wrapper(row):
    """Wrapper for check_antonym_is_token_prefix function"""
    return check_antonym_is_token_prefix(str(row.token), str(row.pos))

def check_antonym_suffix_wrapper(row):
    """Wrapper for check_antonym_is_token_suffix function"""
    return check_antonym_is_token_suffix(str(row.token), str(row.pos))

def find_antonyms(token, pos):
    """Finds antonyms of an input token with help of input PoS tag, returns list of antonyms"""
    lemma = nltk_lemmatizer_token(token, pos=pos)
    antonyms_list = []
    for syn in wordnet.synsets(lemma):
        for l in syn.lemmas():
            if l.antonyms():
                antonyms_list.append(l.antonyms()[0].name())
    return antonyms_list

def check_antonym_is_token_prefix(token, pos):
    """Checks whether antonym is a token without the prefix.
    Input token and PoS-tag, outputs boolean."""
    antonyms_list = find_antonyms(token, pos)
    if len(antonyms_list) == 0:
        return False
    for prefix in prefixes:
        if token.startswith(prefix):
            new_token = token.strip(prefix)
            for antonym in antonyms_list:
                if new_token == antonym:
                    return True
    return False

def check_antonym_is_token_suffix(token, pos):
    """Checks whether antonym is a token without the suffix.
    Input token and PoS-tag, outputs boolean."""
    antonyms_list = find_antonyms(token, pos)
    if len(antonyms_list) == 0:
        return False
    for suffix in suffixes:
        if token.endswith(suffix):
            new_token = token.strip(suffix)
            for antonym in antonyms_list:
                if new_token == antonym:
                    return True
    return False

def check_prefix(token):
    """Checks whether a prefix is in a token.
    Inputs token, outputs boolean."""
    token = token.lower()
    for prefix in prefixes:
        if token.startswith(prefix) and token != prefix  and token not in new_stopwords_list:
            return True
    return False

def check_suffix(token):
    """Checks whether a suffix is in a token.
    Inputs token, outputs boolean."""
    token = token.lower()
    for suffix in suffixes:
        if token.endswith(suffix) and token != suffix and token not in new_stopwords_list:
            return True
    return False

def check_negation_prefix(token):
    """Checks whether the prefix is a negation by looking at the remaining token.
    Inputs token, outputs boolean."""
    new_token = delete_one_prefix(token, prefixes)
    boolean = check_if_word(new_token)
    return boolean

def check_negation_suffix(token):
    """Checks whether the suffix is a negation by looking at the remaining token.
    Inputs token, outputs boolean."""
    new_token = delete_one_suffix(token, suffixes)
    boolean = check_if_word(new_token)
    return boolean

def check_if_word(word):
    """Using NLTK words corpus, a token is checked if it is a word. The output is a boolean"""
    word = word.lower()
    if  1 <= len(word) <= 3:
        return False
    else:
        return word in words.words()
