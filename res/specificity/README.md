model from https://github.com/wjko2/Domain-Agnostic-Sentence-Specificity-Prediction

Only two files have to be modified to contain the data to be evaluated:

'dataset/data/my_data_test.txt'

and

'dataset/data/my_data_unlabeled.txt'

the files can contain the same data, a sentence per line.

the calls for training and predicting should be:

python3 train.py --gpu_id 0 --test_data my_data

python3 test.py --gpu_id 0 --test_data my_data

the modifications that were done to run in my data were in data2.py and train.py/test.py and are documented here:

https://github.com/wjko2/Domain-Agnostic-Sentence-Specificity-Prediction/issues/4

the results scores are at predictions.txt
