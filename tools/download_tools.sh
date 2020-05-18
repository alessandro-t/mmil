#!/bin/bash

# DOWNLOAD CORANLP
wget http://nlp.stanford.edu/software/stanford-corenlp-latest.zip
unzip stanford-corenlp-latest.zip
rm -rf stanford-corenlp-latest.zip 

# DOWNLOAD GLOVE
git clone http://github.com/stanfordnlp/glove
cd glove && make
