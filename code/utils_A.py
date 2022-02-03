import nltk
from nltk import pos_tag
import pandas as pd
import spacy
from nltk.corpus import stopwords, wordnet, words, treebank
from spacy.lang.en.stop_words import STOP_WORDS

nlp = spacy.load("en_core_web_sm", disable=["tokenizer", "parser","ner", "textcat", "custom"])
df = pd.read_csv('./../data/SEM-2012-SharedTask-CD-SCO-training-simple.v2.txt', sep="\t", names=["story", "sent_index", "token_index", "token", "bio"])


multiword_expressions = [["by", "no", "means"], ["on", "the", "contrary"], ["rather", "than"], ["not", "for", "the", "world"], ["nothing", "at", "all"], ["no", "more"]]

negation_stopwords_set = {"n't", "n’t", "n‘t","cannot", "nobody", "neither", "nowhere", "not", "no", "without", "nevertheless", "nor", "never"}
# removing negation words from spacy stopwords set
new_stopwords_list = []
for i in STOP_WORDS:
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

def spacy_tagger_wrapper(row):
    """Wrapper for spacy pos-tag function"""
    return spacy_tagger(str(row.token))

def spacy_tagger(token):
    """Function pos-tags token"""
    doc = nlp(token)
    for i in doc:
        tag = i.pos_
    return tag

def delete_one_prefix(token, prefixes):
    """The prefixes are deleted from an input token, using the prefixes-set. Output is the remaining token"""
    for prefix in prefixes:
        token = token.lower()
        if token.startswith(prefix) and token not in new_stopwords_list:
            token = token.replace(prefix, "")
    return token

def delete_one_suffix(token, suffixes):
    """The suffixes are deleted from an input token, using the suffixes-set. Output is the remaining token"""
    for suffix in suffixes:
        token = token.lower()
        if token.endswith(suffix) or token not in new_stopwords_list:
            token = token.replace(suffix, "")
    return token

def lemma_wrapper(row):
    """Wrapper for lemmatizer function"""
    return spacy_lemmatizer_token(str(row.token))

def spacy_lemmatizer_token(token):
    """One input token is lemmatized using SpaCy. The lemma is returned"""
    doc = nlp(token)
    for i in doc:
        lemma = i.lemma_
    return lemma

def find_antonyms_wrapper(row):
    """Wrapper for find_antonyms function"""
    return find_antonyms(str(row.token))

def check_antonym_prefix_wrapper(row):
    """Wrapper for check_antonym_is_token_prefix function"""
    return check_antonym_is_token_prefix(str(row.token))

def check_antonym_suffix_wrapper(row):
    """Wrapper for check_antonym_is_token_suffix function"""
    return check_antonym_is_token_suffix(str(row.token))

def find_antonyms(token):
    """Finds antonyms of an input token with help of input PoS tag, returns list of antonyms"""
    lemma = spacy_lemmatizer_token(token)
    antonyms_list = []
    for syn in wordnet.synsets(lemma):
        for l in syn.lemmas():
            if l.antonyms():
                antonyms_list.append(l.antonyms()[0].name())
    return antonyms_list

def check_antonym_is_token_prefix(token):
    """Checks whether antonym is a token without the prefix.
    Input token and PoS-tag, outputs boolean."""
    antonyms_list = find_antonyms(token)
    if len(antonyms_list) == 0:
        return False
    for prefix in prefixes:
        if token.startswith(prefix):
            new_token = token.replace(prefix, "")
            for antonym in antonyms_list:
                if antonym.startswith(new_token):
                    return True
    return False

def check_antonym_is_token_suffix(token):
    """Checks whether antonym is a token without the suffix.
    Input token and PoS-tag, outputs boolean."""
    antonyms_list = find_antonyms(token)
    if len(antonyms_list) == 0:
        return False
    for suffix in suffixes:
        if token.endswith(suffix):
            new_token = token.replace(suffix, "")
            for antonym in antonyms_list:
                if antonym.startswith(new_token):
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
