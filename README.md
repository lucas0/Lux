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

    -- unzip uncased_L-12_H-768_A-12.zip

   You should have 3 files, the model, bert_config.json and vocab.txt.

4)Set env variable BERT_BASE_DIR:

    -- export BERT_BASE_DIR=/path/to/Lux/res/bert/uncased_L-12_H-768_A-12

   in our case: export BERT_BASE_DIR=~/Lux/res/bert/uncased_L-12_H-768_A-12

# Install Specificity model

1)Download DASSP.zip inside res/specificity
    
    -- wget https://www.dropbox.com/s/41uw7wm2bbgoff4/DASSP.zip
   
2)Unzip its contents

    -- unzip DASSP.zip

3)Go into folder, download and unzip glove:

    -- cd Domain-Agnostic-Sentence-Specificity-Prediction/
    -- wget https://www.dropbox.com/s/0g880op64chjw4b/glove.840B.300d.zip
    -- unzip glove.840B.300d.zip

+)Check the README.md inside the folder, if modifications have to be done

# Create an virtual environment with python3 and activate it

1)Back to Lux

    -- virtualenv envLux-p python3
    
    -- source envLux/bin/activate
    
# Install requirements

    -- pip install -r requirements.txt

# Download and extract GoogleNews-vectors-negative300.bin into data/

    -- cd data/
    -- wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
    -- gunzip GoogleNews-vectors-negative300.bin.gz

## Known issues

semantic complexity might raise a division by 0 critical error if the texts fed are not big enough. A simple solution to that is increasing the minimum lenght of body texts.
