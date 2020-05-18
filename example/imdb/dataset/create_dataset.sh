#!/bin/bash

# This script calls several subroutines for preprocessing the raw 
# IMDB dataset.

# When all the subroutines are executed the final imdb dataset will be saved
# in data/imdb.npy

mkdir -p data
python download.py

# the integer represents the number of processes
python preprocess.py 4

# the float represents the proportion for the validation set
# the integer represents the seed 
# NOTE: The repository already includes the validation_set.txt file.
# The validation set may change on different version of python even
# when the same seed is used. Uncomment the following line if you prefer
# to create your own validation set.
#python extract_validation.py 0.1 1234

python pkl2glove.py

# Train GloVe
set -e 
mkdir -p data/glove_output
CORPUS=data/imdb_for_glove.txt
VOCAB_FILE='data/glove_output/vocab.txt'
COOCCURRENCE_FILE='data/glove_output/cooccurrence.bin'
COOCCURRENCE_SHUF_FILE='data/glove_output/cooccurrence.shuf.bin'
BUILDDIR='../../../tools/glove/build'
SAVE_FILE='data/glove_output/vectors'
VERBOSE=2
MEMORY=4.0
VOCAB_MIN_COUNT=2
VECTOR_SIZE=100
MAX_ITER=15
WINDOW_SIZE=15
BINARY=2
NUM_THREADS=8
X_MAX=10

echo
echo "$ $BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE"
$BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE
echo "$ $BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE"
$BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE
echo "$ $BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE"
$BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE
echo "$ $BUILDDIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE"
$BUILDDIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE

python pack_all.py
