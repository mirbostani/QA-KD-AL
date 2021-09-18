#!/usr/bin/env bash

# GloVe
# https://nlp.stanford.edu/projects/glove/
GLOVE_DIR=./data/glove
mkdir -p "$GLOVE_DIR"
wget -c "https://nlp.stanford.edu/data/glove.840B.300d.zip" -O "$GLOVE_DIR/glove.840B.300d.zip"
unzip "$GLOVE_DIR/glove.840B.300d.zip" -d "$GLOVE_DIR"
wget -c "https://raw.githubusercontent.com/minimaxir/char-embeddings/master/glove.840B.300d-char.txt" -O "$GLOVE_DIR/glove.840B.300d-char.txt"

# SQuAD v1.1
SQUAD_DIR=./data/squad
mkdir -p "$SQUAD_DIR"
wget -c "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json" -O "$SQUAD_DIR/train-v1.1.json"
wget -c "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json" -O "$SQUAD_DIR/dev-v1.1.json"

# Adversarial SQuAD
# https://worksheets.codalab.org/worksheets/0xc86d3ebe69a3427d91f9aaa63f7d1e7d/
ADV_SQUAD_DIR=./data/adv_squad
mkdir -p "$ADV_SQUAD_DIR"
wget -c "https://worksheets.codalab.org/rest/bundles/0xb765680b60c64d088f5daccac08b3905/contents/blob/" -O "$ADV_SQUAD_DIR/sample1k-HCVerifyAll_AddSent.json"
wget -c "https://worksheets.codalab.org/rest/bundles/0x3ac9349d16ba4e7bb9b5920e3b1af393/contents/blob/" -O "$ADV_SQUAD_DIR/sample1k-HCVerifySample_AddOneSent.json"

# Download Spacy English Language
# conda activate
python -m spacy download en