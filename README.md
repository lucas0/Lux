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
   
5)Start bert-as-a-service server for requests in another session/screen tab

    -- bert-serving-start -model_dir $BERT_BASE_DIR -max_seq_len 512 -mask_cls_sep

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

# Running

    -- bash run.sh

    OR

    -- sudo -E python3 lux.py

if 'True' is passed as first argument, force_reload will receive its value and new bert models as well as new features will be generated.

---

# Papers

Please cite the published articles related to this work:

Azevedo, Lucas, et al. "LUX (Linguistic aspects Under eXamination): Discourse Analysis for Automatic Fake News Classification." Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021. 2021.

    @inproceedings{azevedo2021lux,
      title={LUX (Linguistic aspects Under eXamination): Discourse Analysis for Automatic Fake News Classification},
      author={Azevedo, Lucas and d’Aquin, Mathieu and Davis, Brian and Zarrouk, Manel},
      booktitle={Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021},
      pages={41--56},
      year={2021}
    }

Azevedo, Lucas, and Mohamed Moustafa. "Veritas annotator: Discovering the origin of a rumour." Proceedings of the Second Workshop on Fact Extraction and VERification (FEVER). 2019.

    @inproceedings{azevedo2019veritas,
      title={Veritas annotator: Discovering the origin of a rumour},
      author={Azevedo, Lucas and Moustafa, Mohamed},
      booktitle={Proceedings of the Second Workshop on Fact Extraction and VERification (FEVER)},
      pages={90--98},
      year={2019}
    }

Azevedo, Lucas. "Truth or lie: Automatically fact checking news." Companion Proceedings of the The Web Conference 2018. 2018.

    @inproceedings{azevedo2018truth,
      title={Truth or lie: Automatically fact checking news},
      author={Azevedo, Lucas},
      booktitle={Companion Proceedings of the The Web Conference 2018},
      pages={807--811},
      year={2018}
    }
