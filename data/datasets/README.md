After consolidating the annotation with goldFilter.py into, for example, datasetVeritas3.csv (at the annotator folder), there is still the origin crawling that has to be done to generate the file with the right header to be processed by the lux model.

input_data is the concatenation of the other files:

good_a3.csv is the same file that is in the repo VeritasCorpus/goldDatav2.1
pos_samples IDEM

veritas v3.0 contains the good results of the FIRST manual annotation, 168 examples that had at > 0  "yes" vote 

veritas v4.0 contains the good results of the MAIN manual annotation, 722 examples that had at > 0  "yes" vote  

concat+normalize.py, as the name indicates, concatenates all those files and normalize them, saving to dataset.csv

on the data_loader.py, the dataset.csv file will be read, shuffled with a seed code and saved to data/data.csv, a hash will be generated from that dataframe to allow for checking and skipping of processing steps.
