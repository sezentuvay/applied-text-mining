PROJECT APPLIED TEXT MINING 1: METHODS
GROUP-TM-2
2022

AUTHORS
-------
Ajda Efendi, Ellemijn Galjaard, Sezen Tuvay, Sybren Moolhuizen

TABLE OF CONTENTS
-------------------
```
├── annotations             # includes annotations on a different corpus as an experiment             
│   ├── Ajda_annotations-20220115T130852Z-001.zip
│   ├── Ellemijn_annotations-20220116T002456Z-001.zip
│   ├── Sezen_annotations-20220115T131415Z-001.zip
│   └── Sybren_annotations-20220115T131446Z-001.zip              
├── code
│   ├── main_A.py           # spaCy pipeline
│   ├── main_B.py           # NLTK pipeline
│   ├── utils_A.py          # utils for spaCy pipeline
│   ├── utils_B.py          # utils for NLTK pipeline
│   ├── SVM.py              # code for running the SVM algorithm
│   ├── CRF.py              # code for running the CRF algorithm
│   └── NN.py               # code for running the MLP neural network
├── data
│   ├── SEM-2012-SharedTask-CD-SCO-training-simple.v2.txt
│   ├── SEM-2012-SharedTask-CD-SCO-dev-simple.v2.txt
│   ├── SEM-2012-SharedTask-CD-SCO-test-cardboard.txt
│   └── SEM-2012-SharedTask-CD-SCO-test-circle.txt
├──  results
│   ├── train_features.tsv   # features training file  
│   └── dev_features.tsv     # features development file
│   └── test_features.tsv    # features test file (both test files combined into one)
```


PROJECT DESCRIPTION
-------------------
This project is about detecting negation cues in text. The project idea and data are taken from the \*SEM Shared Task 2012.

Inside the "code" folder, there are several files to obtain additional features from the text and run several different algorithms.
The project enables you to train multiple classification algorithms and combinations of features to find and label the negation cues in the data.
The purpose of the classifiers is to ascribe each token a B-NEG (beginning of negation cue), I-NEG (inside negation cue, in case of multiword negations) or O (non-negation) label. 

After conducting an error analysis, the classifier that worked the best proved to be the SVM algorithm with the following features: *token, token-1, token-2, token+1, token+2, lemma, pos, chunk, matchesNeg, hasPrefix, hasPrefixAntonym* and *matchesMulticue*. This classifier is recommended.

HOW TO OPERATE THE PROGRAM
-------------------------
Download the code from GitHub, unzip the file and navigate to the relevant directory in the terminal with the 'cd' command.

Run either the main_A.py file (the spaCy pipeline) or the main_B.py file (the NLTK pipeline) to obtain the features for each of the files.
#### Pipeline B is recommended!

Before running the code, it is recommended to create a new virtual environment and run **""pip install -r requirements.txt"** to get the specific packages needed.

To obtain the features from the terminal:
1. First run **"python main_B.py train"** to output the training file with all added features in the 'results' directory.
2. Next, run **"python main_B.py dev"** to output the development file with all added features in the 'results' directory.
3. Next, run **"python main_B.py test"** to output the test file (/combined test files) with all added features in the 'results' directory.

To run the algorithms from the terminal:
1. Run either **"python SVM.py dev"** or **"python SVM.py test"** to run the SVM algorithm and print a classification report.
2. Run either **"python CRF.py dev"** or **"python CRF.py test"** to run the CRF algorithm and print a classification report.
    - Please do not be intimidated by one of many futurewarnings that may rise.

There is also an option to run a neural network by running **"python NN.py"** in the terminal. This neural network is meant as an additional experiment and tests its performance using scikit-learn's train_test_split() function. **It is not to be compared with the SVM and CRF algorithms!**

Inside the algorithm files, features can be excluded by commenting them out.

LICENSE (Copied from MIT)
-------------------------
Copyright (c) [2022]
[Ajda Efendi, Ellemijn Galjaard, Sezen Tuvay, Sybren Moolhuizen]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

