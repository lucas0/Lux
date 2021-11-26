from pathos.multiprocessing import ProcessingPool as PathosPool
import multiprocessing as mp
import seaborn as sns
import numpy as np
from multiprocessing import Pool
import docker
import time
import unicodedata
import itertools
from pickle import dump, load
from itertools import dropwhile
import re
from lexical_diversity import lex_div as ld
from uncertainty.classifier import Classifier
import readability
import subprocess
import pandas as pd
import swifter
import os, sys
from textblob import TextBlob
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
cwd = os.path.abspath(os.path.dirname(sys.argv[0]))
data_dir = cwd+"/data"
spec_dir = cwd+"/res/specificity/Domain-Agnostic-Sentence-Specificity-Prediction"
comp_dir = cwd+"/res/complexity"
read_dir = cwd+"/res/readability"
res_path = cwd+"/res"

pos_list={'CC':0,'CD':0,'DT':0,'EX':0,'FW':0,'IN':0,'JJ':0,'JJR':0,'JJS':0,'LS':0,'MD':0,'NN':0,'NNS':0,'NNP':0,'NNPS':0,'PDT':0,'POS':0,'PRP':0,'PRP$':0,'RB':0,'RBR':0,'RBS':0,'RP':0,'SYM':0,'TO':0,'UH':0,'VB':0,'VBD':0,'VBG':0,'VBN':0,'VBP':0,'VBZ':0,'WDT':0,'WP':0,'WP$':0,'WRB':0, ',':0, '.':0, '(':0, ')':0, '$':0, '\'\'':0, '``':0, ':':0, '#':0}

text = "The cat in the hat barely knows something"
text2 = "Pizza is the best food ever"
text = "Results show that the sky is blue blue blue"

alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"
digits   = "([0-9])"

def split_into_sentences(text):
        text = " " + text + "  "
        text = text.replace("\n"," ")
        text = text.replace("``","\"")
        text = text.replace("`","'")
        text = text.replace("\'\'","\"")
        text = re.sub(prefixes,"\\1<prd>",text)
        text = re.sub(websites,"<prd>\\1",text)
        text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
        if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
        text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
        text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
        text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
        text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
        text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
        text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
        text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
        if "”" in text: text = text.replace(".”","”.")
        if "\"" in text: text = text.replace(".\"","\".")
        if "!" in text: text = text.replace("!\"","\"!")
        if "?" in text: text = text.replace("?\"","\"?")
        text = text.replace(".",".<stop>")
        text = text.replace("?","?<stop>")
        text = text.replace("!","!<stop>")
        text = text.replace("<prd>",".")
        sentences = text.split("<stop>")
        sentences = sentences[:-1]
        sentences = [s.strip() for s in sentences]
        return sentences

log_filename = cwd+"/feat_log.txt"
def log_results(data_frame, feat_name):
    with open(log_filename, "a+") as f:
        f.write(feat_name+"\n\n")
        f.write(data_frame.to_string(header=True, index=True))

#this process takes a while, so it should be done before the feature generation step
def generate_specificity():
    my_data_path = spec_dir+"/dataset/data/my_data_unlabeled.txt"
    csv = pd.read_csv(data_dir+"/data.csv", sep = "\t")
    csv.body = csv.body.astype('str')
    texts = ["%s\n" % re.sub("\n|\r", "",t) for t in csv['body']]

    #writes sentences to a file
    with open(my_data_path, 'wb') as file:
        #the framework used ignores the first line of the document, so I am duplicating it.
        file.write(texts[0].encode('ascii', 'ignore'))
        for line in texts:
            file.write(line.encode('ascii', "ignore"))

    subprocess.call("cp "+my_data_path+" "+spec_dir+"/dataset/data/my_data_test.txt", shell=True, cwd = spec_dir)
    #calls subprocess to get sentences scores
    subprocess.call("python3 train.py --gpu_id 0 --test_data my_data", shell=True, cwd = spec_dir)
    subprocess.call("python3 test.py --gpu_id 0 --test_data my_data", shell=True, cwd = spec_dir)
    print("Specificity Scores Generated")

def generate_complexity():
    csv = pd.read_csv(data_dir+"/data.csv", sep="\t")
    csv.body = csv.body.astype('str')
    print("Generating Complexity for dataset shaped: ",csv.shape)
    subprocess.call("rm -f "+cwd+"/res/complexity/input_texts/*", shell=True, cwd=comp_dir)

    with mp.Pool() as pool:
        pool.map(complexity_aux, [(e['body'],str(idx)) for idx,e in csv.iterrows()])

    client = docker.from_env()
    docker_containers = client.containers.list(all=True)
    for dc in docker_containers:
        dc.remove(force=True)
    container = client.containers.run("dbpedia/spotlight-english:databus", "spotlight.sh", detach=True, restart_policy={"Name": "on-failure", "MaximumRetryCount": 900000}, ports={'80': 2222})
    time.sleep(75)

    try:
        subprocess.call("python3 pysemcom.py texts2vectors -nc 8 input_texts/ output_file.csv http://0.0.0.0:2222/rest/annotate", shell=True, cwd=comp_dir)
    except subprocess.CalledProcessError as e:
        print(traceback.format_exc())
        input("ERROR during complexity scores generation")
        return

    finally:
        container.stop()

    print("Complexity Scores Generated")

def readability_aux(t):
    body_text, index = t
    with open(cwd+"/res/readability/input_texts/"+index+".txt", "w+", encoding="utf-8") as f:
        text = re.sub("\n|\r","",body_text)
        f.write(body_text)


#this process takes a while, so it should be done before the feature generation step
def complexity_aux(t):
    body_text, index = t
    with open(cwd+"/res/complexity/input_texts/"+index+".txt", "w+", encoding="utf-8") as f:
        text = re.sub("\n|\r","",body_text)
        f.write(text)

analyzer = SentimentIntensityAnalyzer()
def affect_aux(t):
    text, sentences, n_sentences, blob = t
    pos_blob, neg_blob = [],[]

    for s in blob.sentences:
       p = s.sentiment.polarity
       pos_blob.append(p) if p > 0 else neg_blob.append(p)

    avg_blob = sum(pos_blob+neg_blob) / len(blob.sentences)
    pos_blob = sum(pos_blob) / len(pos_blob) if len(pos_blob) > 0 else 0
    neg_blob = sum(neg_blob) / len(neg_blob) if len(neg_blob) > 0 else 0
    pos_vader, neg_vader = [],[]

    for s in sentences:
       p = analyzer.polarity_scores(s)['compound']
       pos_vader.append(p) if p > 0 else neg_vader.append(p)

    avg_vader = sum(pos_vader+neg_vader) / n_sentences
    pos_vader = sum(pos_vader) / len(pos_vader) if len(pos_vader) > 0 else 0
    neg_vader = sum(neg_vader) / len(neg_vader) if len(neg_vader) > 0 else 0

    return avg_blob,pos_blob,neg_blob,avg_vader,pos_vader,neg_vader

#sets the passiveness tagger up
if os.path.exists("tagger.pkl"):
   with open('res/brown_tagger.pkl', 'rb') as data:
      tagger = load(data)
else:
   from nltk.corpus import brown
   train_sents = brown.tagged_sents()
   t0 = nltk.RegexpTagger(
      [(r'^-?[0-9]+(.[0-9]+)?$', 'CD'), # cardinal numbers
      (r'(The|the|A|a|An|an)$', 'AT'), # articles
      (r'.*able$', 'JJ'),              # adjectives
      (r'.*ness$', 'NN'),              # nouns formed from adjectives
      (r'.*ly$', 'RB'),                # adverbs
      (r'.*s$', 'NNS'),                # plural nouns
      (r'.*ing$', 'VBG'),              # gerunds
      (r'.*ed$', 'VBD'),               # past tense verbs
      (r'.*', 'NN')                    # nouns (default)
      ])
   t1 = nltk.UnigramTagger(train_sents, backoff=t0)
   t2 = nltk.BigramTagger(train_sents, backoff=t1)
   tagger = nltk.TrigramTagger(train_sents, backoff=t2)
   with open('res/brown_tagger.pkl', 'wb') as output:
      dump(tagger, output, -1)

def passiveness_aux(t):
    sentences, n_sent = t
    count = 0
    for s in sentences:
        tagged = tagger.tag(nltk.word_tokenize(s))
        tags = map(lambda tup: tup[1], tagged)
        after_to_be = list(dropwhile(lambda tag: not tag.startswith("BE"), tags))
        nongerund = lambda tag: tag.startswith("V") and not tag.startswith("VBG")
        filtered = filter(nongerund, after_to_be)
        out = any(filtered)
        if bool(out): count+=1

    return(count/n_sent)

def generateFeats(feature_list=["inf","div","qua","aff","sbj","spe","pau","unc","pas"]):
    csv = pd.read_csv(data_dir+"/data.csv", sep="\t")
    print("Generating Features for dataset shaped: ",csv.shape)

    csv['body'] = csv['body'].astype('str')
    csv['sent'] = csv['body'].swifter.progress_bar(desc="-Splitting into sentences...").apply(split_into_sentences).astype('str')
    csv['sent'].str.lower()
    csv['n_sent'] = csv['sent'].swifter.progress_bar(desc="-Counting number of sentences...").apply(lambda x: max(len(x),1))

    csv['tokens'] = csv['body'].swifter.progress_bar(desc="-Tokenizing...").apply(nltk.word_tokenize)
    csv['n_tokens'] = csv['tokens'].swifter.progress_bar(desc="-Counting number of tokens...").apply(len)
    csv['tagged'] = csv['tokens'].swifter.progress_bar(desc="-P.O.S. Tagging...").apply(nltk.pos_tag)
    csv['blob'] = csv['body'].swifter.progress_bar(desc="-Enriching with Blob...").apply(TextBlob)

    ### SUBJECTIVITY ###
    if "sbj" in feature_list:
        print("Generating subjectivity scores")
        def subjectivity_lex_aux(tagged_tokens, lex):
            score = 0
            for idx, token in enumerate(tagged_tokens):
                same_pos = lex[(lex['word1'] == token[0]) & (lex['pos1'] == token[1][:2])]
                any_pos = lex[(lex['word1'] == token[0]) & (lex['pos1'] == 'ANY')]

                if not same_pos.empty:
                    score += same_pos.priorpolarity.values[0]
                elif not any_pos.empty:
                    score += any_pos.priorpolarity.values[0]

            return score

        #blob score
        blob_subj_scores = csv['blob'].swifter.progress_bar(desc="-Getting Blob subjectivity scores...").apply(lambda x: x.sentiment.subjectivity)

        #lexicon score
        mpqa_path = res_path+"/subjectivity/MPQA/"
        lex = pd.read_csv(mpqa_path+"/lexicon.csv", sep=',')

        subjectivity_scores = [list(a) for a in zip(csv['tagged'].swifter.progress_bar(desc="-Getting Lexicons subjectivity scores...").apply(lambda x: subjectivity_lex_aux(x,lex))/csv['n_tokens'], blob_subj_scores)]
        sbj = pd.DataFrame(subjectivity_scores, columns=["MPQA subjectivity score over sentences","Blob's Subjectivity over sentences"])

    ### SPECIFICITY ###
    if "spe" in feature_list:
        print("Generating specificity scores")
        with open(spec_dir+"/predictions.txt", 'r+') as file:
            lines = file.readlines()

        spec_scores = pd.DataFrame(lines, columns=['preds'])
        specificity_scores = spec_scores['preds'].swifter.progress_bar(desc="-Getting Specificity scores from DASSP...").apply(lambda x: float(x.split('(')[1].split(',')[0]))
        spe = pd.DataFrame(specificity_scores.to_list(), columns=["Speciteller's scores"])

    ### PAUSALITY ###
    if "pau" in feature_list:
        print("Generating pausality scores")
        pausality_scores = csv['tagged'].swifter.progress_bar(desc="-Getting Pausality scores...").apply(lambda tagged_tokens: sum([1 for word,tag in tagged_tokens if tag == '.']))/csv['n_tokens']
        pau = pd.DataFrame(pausality_scores.to_list(), columns=["#of '.'-tagged tokens"])

    ### INFORMALITY ####
    if "inf" in feature_list:
        print("Generating informality scores")
        #readability
        subprocess.call("rm -f "+cwd+"/res/readability/input_texts/*", shell=True, cwd=read_dir)
        num_of_digits = len(str(len(csv)))
        with mp.Pool() as pool:
            pool.map(readability_aux, [(e['body'],str(idx).zfill(num_of_digits)) for idx,e in csv.iterrows()])
            pool.map(complexity_aux, [(e['body'],str(idx)) for idx,e in csv.iterrows()])

        subprocess.call("readability --csv  "+cwd+"/res/readability/input_texts/* > "+comp_dir+"/readabilitymeasures.csv", shell=True, cwd=comp_dir)
        read_scores = pd.read_csv(comp_dir+"/readabilitymeasures.csv").iloc[:,range(1,10)]

        #complexity
        compl_scores = pd.read_csv(cwd+"/res/complexity/output_file.csv")
        compl_scores = compl_scores.drop(columns="filename")
        compl_scores = pd.read_csv(cwd+"/res/complexity/output_file.csv").iloc[:,1:]
        informality_scores = pd.concat([read_scores, compl_scores], axis=1)
        inf = informality_scores

    ### DIVERISTY ###
    if "div" in feature_list:
        print("Generating diversity scores")
        lemmatized_text = csv['body'].swifter.progress_bar(desc="-Lemmatizing...").apply(lambda x: ld.flemmatize(x))

        def diversity_aux(text):
            funcs = [ld.ttr,ld.root_ttr,ld.log_ttr,ld.maas_ttr,ld.msttr,ld.mattr,ld.hdd,ld.mtld,ld.mtld_ma_wrap,ld.mtld_ma_bid]
            return list(map(lambda x: x(text), funcs))

        diversity_scores = lemmatized_text.swifter.progress_bar(desc="-Getting Diversity scores from Lexical Diversity...").apply(diversity_aux)
        div = pd.DataFrame(diversity_scores.to_list(), columns=["simple TTR", "root TTR", "log TTR", "Mass TTR", "MSTTR", "MATTR", "HDD", "MLTD", "MA-warp", "MA-biD"])

    ### QUANTITY ###
    if "qua" in feature_list:
        print("Generating quantity scores")
        #takes the tagged text as input and outputs simple counts
        def quantity_aux(tagged_tokens):
            pos_counts = pos_list
            for word, tag in tagged_tokens:
                pos_counts[tag] = pos_counts[tag] + 1
            num_terms, num_tokens = len(set(tagged_tokens)), len(tagged_tokens)
            counts = [value/num_terms for key,value in sorted(pos_counts.items())]

            return [num_terms, num_tokens]+counts

        quantity_scores = csv['tagged'].swifter.progress_bar(desc="-Getting Quantity scores...").apply(quantity_aux)
        qua = pd.DataFrame(quantity_scores.to_list(), columns=["#terms", "#tokens", "#", "$", "\"", "(", "''", "''", ".", ":", "CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NN", "NNP", "NNPS", "NNS", "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP", "WP$", "WRB", "``"])

    ### UNCERTAINTY ###
    if "unc" in feature_list:
        print("Generating uncertainty scores")
        #cls = Classifier(granularity=True)
        cls = Classifier()
        def uncertainty_aux(text):
            results = cls.predict(text)
            unc_ratio = sum([1 for r in results if r == 'U'])/len(results)

            return unc_ratio

        uncertainty_scores = csv['body'].swifter.progress_bar(desc="-Getting Uncertainty scores...").apply(uncertainty_aux)
        unc = pd.DataFrame(uncertainty_scores.to_list(), columns=["LUCI score"])

    ### AFFECT ###
    if "aff" in feature_list:
        print("Generating affect scores")
        #takes the raw text as input, and calculates polarity as the average polarity over the sentences
        #for vader, takes the text split in sentences
        p = PathosPool(16)
        affect_scores = pd.DataFrame(p.map(affect_aux, [(e['body'],e['sent'],e['n_sent'],e['blob']) for _,e in csv.iterrows()]), columns=["BPol-avg","BPol-pos", "BPol-neg", "VPol-avg", "VPol-pos", "VPol-neg"])
        aff = affect_scores

    ### PASSIVENESS ###
    if "pas" in feature_list:
        print("Generating passiveness scores")
        with mp.Pool() as pool:
            passiveness_scores = pool.map(passiveness_aux, [(e['sent'],e['n_sent']) for _,e in csv.iterrows()])
        pas = pd.DataFrame(passiveness_scores, columns=["Passiveness score"])

    ### CONCATENATION ###
    print("Concatenating scores")
    concat = []
    for v,f_name in zip([inf,div,qua,aff,sbj,spe,pau,unc,pas], ["inf","div","qua","aff","sbj","spe","pau","unc","pas"]):
        if f_name in feature_list:
            concat.append(v)
    #features = pd.concat([inf,div,qua,aff,sbj,spe,pau,unc,pas], axis=1)
    features = pd.concat(concat, axis=1)
    #features = pd.concat([inf,div,aff,sbj,spe,pau,unc,pas], axis=1)
    print("Features Generated. Shaped: ",features.shape)
    print("Calculating correlation. Saving to results/correlation/")

    # calculate the correlation matrix
    #corr = features.corr()

    # plot the heatmap
    #sns.set(rc={'figure.figsize':(20,18)})
    #sns.set(font_scale=0.2)
    #sns_plot = sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)

    # saves the heatmap to file
    #fig = sns_plot.get_figure()
    #fig.savefig(cwd+"/results/correlation/heatmap.png")
    #print("correlation image saved.")

    return features.to_numpy().astype(np.float)
