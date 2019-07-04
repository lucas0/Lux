## Model for text classification regarding veracity.

Baseline uses W2V embeddings trained on google corpus for fake news

Lux proposes the usage of Linguistic Aspects as Features.
# Lux

## INSTALLATION

# This repository uses bert. in order to use BERT properly:

1)Clone bert repo inside Lux/res:

    -- git clone https://github.com/google-research/bert

2)Download the pre-trained model from bert:

    -- wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip

3)Unzip the model inside bert folder:

   -- unzip uncased_L-12_H-768_A-12.zip.

   You should have 3 files, the model, bert_config.json and vocab.txt.

4)Set env variable BERT_BASE_DIR:

   -- export BERT_BASE_DIR=/path/to/Lux/res/bert/uncased_L-12_H-768_A-12

   in our case: export BERT_BASE_DIR=~/Lux/res/bert/uncased_L-12_H-768_A-12

# Create an virtual environment with python3 and activate it

    --virtualenv envLux -p python3

    --source envLux/bin/activate

# Install requirements

    --pip install -r requirements.txt

# Download GoogleNews-vectors-negative300.bin

    -- Download file and 'gunzip GoogleNews-vectors-negative300.bin.gz' inside data/
