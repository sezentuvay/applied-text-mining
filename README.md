PROJECT APPLIED TEXT MINING 1: METHODS
GROUP-TM-2
2022

AUTHORS
-------
Ajda Efendi, Ellemijn Galjaard, Sezen Tuvay, Sybren Moolhuizen

PROJECT DESCRIPTION
-------------------
This project is about training multiple classification algorithms and combinations
of features to find and label negation cues in text. It was the shared task of the
*SEM conference of 2012.

After creating and testing various combinations of features and hyperparameters,
a best performing system was used to perform an error analysis. The best performing
algorithm was also used to predict a label based on the test set.

TABLE OF CONTENTS
-------------------
├── annotations                  
│   ├── Ajda_annotations-20220115T130852Z-001.zip
│   ├── Ellemijn_annotations-20220116T002456Z-001.zip
│   ├── Sezen_annotations-20220115T131415Z-001.zip
│   └── Sybren_annotations-20220115T131446Z-001.zip              
├── code
│   ├── main_A.py       # spaCy pipeline
│   ├── main_B.py       # NLTK pipeline
│   ├── utils_A.py
│   └── utils_B.py
├── data
│   ├── SEM-2012-SharedTask-CD-SCO-dev-simple.v2.txt
│   ├── SEM-2012-SharedTask-CD-SCO-training-simple.v2.txt
│   ├── dev_features_B.tsv
│   └── training_features_B.tsv
├──  results
│   ├── features_A.tsv   # features spaCy pipeline   
│   └── features_B.tsv   # features NLTK pipeline

HOW TO OPERATE THE PROGRAM
-------------------------
Inside of the "code" folder there are python (.py) files for the separates
pipelines. Running the main_A and main_B files outputs the .tsv filed inside of
the "results" folder.


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

