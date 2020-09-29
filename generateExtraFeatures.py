import requests
from bert_serving.server import BertServer
from bert_serving.server.helper import get_args_parser
from bert_serving.client import BertClient
import re
import subprocess
import json
#import docker
from timeit import default_timer as timer
#import unicodedata
#import itertools
#from pickle import dump, load
#from itertools import dropwhile
#from lexical_diversity import lex_div as ld
#from uncertainty.classifier import Classifier
#import readability
#import subprocess
import pandas as pd
import os, sys
#from textblob import TextBlob
#import nltk
cwd = os.path.abspath(os.path.dirname(sys.argv[0]))
data_dir = cwd+"/data"
spec_dir = cwd+"/res/specificity/Domain-Agnostic-Sentence-Specificity-Prediction"
comp_dir = cwd+"/res/complexity"
res_path = cwd+"/res"
bert_dir = cwd+"/res/bert"

#a_author,a_date,a_tags,a_url,claim,dataset,name,o_author,o_body,o_date,o_domain,o_keywords,o_summary,o_title,o_url,source_list,value,verdict
#a_author,claim,o_author,o_summary,o_title,source_list,

opr_header = {'API-OPR':'8kcosso0gk8sw8g0osckwowk4w8kc0wowg888coo'}
def page_rank(o_domain):
    url = 'https://openpagerank.com/api/v1.0/getPageRank?domains%5B0%5D=' + o_domain
    request = requests.get(url, headers=opr_header)
    result = request.json()
    print(result)
    return(result)

#def generate_tags_bow():
#a_tags
#o_keywords
#
#def dates_and_difference():
#a_date =
#o_date =
#date_dif =
#
#def text_similarity(t1,t2):

def getBERT(texts):

    #start the bert-as-a-service server
    bert_dir = os.environ.get("BERT_BASE_DIR")
    args = get_args_parser().parse_args(['-model_dir', bert_dir, '-port', '5555', '-port_out', '5556', '-max_seq_len', 'NONE', '-mask_cls_sep'])
    server = BertServer(args)
    server.start()

    #generates the encodings for the texts
    bc = BertClient(check_version=False)
    for text in texts:
        a = bc.encode([text])

    #stops the bert-as-a-service server
    #bc.shutdown(port=5555)


df = pd.read_csv(data_dir+"/data.csv", sep="\t")
df = df.body.tolist()
print(len(df))

start = timer()
getBERT(df)
end = timer()
print("time",end - start)

def extraOI(ext):
    a_date, a_tags, claim, o_body, o_date, o_domain, o_keywords, o_summary, o_title, source_list = ext
    #1)obtain a_body?
    #2)do similarity between: (a_tags, claim, a_body) X (o_keywords, o_title, o_summary, o_body)
    #3)lenght of source_list
    #4)dates_and_difference

def extraLUX(ext):
    o_body, o_date, o_domain, o_keywords, o_summary, o_title, source_list = ext
    #for extra LUX features
    #should use only origin features
    #1)append title/summary to bert! (data_load.py)
    #2)similarity between origin pieces (to check consistency)
    #3)lenght of source_list
    #4)dates_and_difference (using current date and origin date)


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

#this process takes a while, so it should be done before the feature generation step
def generate_specificity():
    my_data_path = spec_dir+"/dataset/data/my_data_unlabeled.txt"
    csv = pd.read_csv(data_dir+"/data.csv", sep = "\t")
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

#this process takes a while, so it should be done before the feature generation step
def generate_complexity():
    csv = pd.read_csv(data_dir+"/data.csv", sep="\t")
    print("Generating Complexity for dataset shaped: ",csv.shape)
    subprocess.call("rm "+cwd+"/res/complexity/input_texts/*", shell=True, cwd=comp_dir)
    for index, row in csv.iterrows():
        with open(cwd+"/res/complexity/input_texts/"+str(index)+".txt", "w+", encoding="utf-8") as f:
            text = re.sub("\n|\r","",row['body'])
            f.write(text)

    client = docker.from_env()
    container = client.containers.run("dbpedia/spotlight-english:databus", "spotlight.sh", detach=True, restart_policy={"Name": "on-failure", "MaximumRetryCount": 900000}, ports={'80': 2222})
    time.sleep(75)

    try:
        #subprocess.call("python3 pysemcom.py texts2vectors input_texts/ output_file.csv http://api.dbpedia-spotlight.org/en/annotate", shell=True, cwd=comp_dir)
        subprocess.call("python3 pysemcom.py texts2vectors input_texts/ output_file.csv http://0.0.0.0:2222/rest/annotate", shell=True, cwd=comp_dir)
    except subprocess.CalledProcessError as e:
        print(traceback.format_exc())
        input("ERROR during complexity scores generation")
        return

    finally:
        container.stop()

    print("Complexity Scores Generated")

def vectorize(text, text_id, complexity=False, drop_feat_idx=[]):
    print("Vectorizing: ",len(text))
    sentences = split_into_sentences(text)
    lower_case = text.lower()
    tokens = nltk.word_tokenize(lower_case)
    tagged_tokens = nltk.pos_tag(tokens)
    blob = TextBlob(text)
    n_sentences = max([len(sentences),1])

    sbj = subjectivity_features(tagged_tokens, blob)[1]
    spe = specificity_features(text_id)
    pau = pausality_features(tagged_tokens, n_sentences)
    inf = informality_features(text, text_id, complexity=complexity)
    div = diversity_features(text)
    qua = quantity_features(tagged_tokens)
    unc = uncertainty_features(text)
    aff = list(affect_features(text, sentences, n_sentences, blob))

    vector = inf+div+qua+aff
    vector.extend([sbj,spe,pau,unc])
    #removing non deterministic features: Diversity(Hypergeometric Distribution D) and Specificity
    #indexes 40 and 94
    #print("\nvec before and after Diversity\n")
    #print(vector[38:42])
    #print("div:",div)
    #del vector[40]
    #print(vector[38:42])
    #input("confirm right drops")
    vector = [v for i,v in enumerate(vector) if i not in drop_feat_idx]
    #drop_feat_idx
    return vector

#takes the tagged text as input and outputs subjectivity scores
def subjectivity_features(tagged_tokens, blob):
        blob_score = blob.sentiment.subjectivity
        mpqa_path = res_path+"/subjectivity/MPQA/"
        lex = pd.read_csv(mpqa_path+"/lexicon.csv", sep=',')
        score = 0
        for idx, token in enumerate(tagged_tokens):
                same_pos = lex[(lex['word1'] == token[0].lower()) & (lex['pos1'] == token[1][:2])]
                any_pos = lex[(lex['word1'] == token[0].lower()) & (lex['pos1'] == 'ANY')]

                if not same_pos.empty:
                        score += same_pos.type
                elif not any_pos.empty:
                        score += any_pos.type

        return score/len(text),blob_score

#takes sentences as the argument
def specificity_features(text_id):
        with open(spec_dir+"/predictions.txt", 'r+') as file:
                lines = file.readlines()
                score = lines[text_id].split('(')[1]
                score = score.split(',')[0]
                float(score)
        return score

#Takes as input the whole text and the number of sentences. Returns a ratio between the number of punctuation tokens and the number of sentences.
def pausality_features(tagged_tokens, num_sentences):
        punct = sum([1 for word,tag in tagged_tokens if tag == '.'])
        return (punct/num_sentences)

#if complexity is to be used, some data preprocessing would have to be done and here, the scores would only be read
def informality_features(text, text_id=0, complexity=False):
        #for readability, sentences should be separated by '\n'
        results = readability.getmeasures(text, lang='en')
        informality_scores = [score[1] for score in results['readability grades'].items()]

        #reads the semanticComplexity from output_file.txt on line order matters as the id should match the line
        if complexity:
                complexity = pd.read_csv(cwd+"/res/complexity/output_file.csv")
                complexity_score = complexity.iloc[text_id].values.tolist()
                informality_scores.extend(complexity_score)

        return informality_scores

def diversity_features(text):
        flt = ld.flemmatize(text)
        funcs = [ld.ttr,ld.root_ttr,ld.log_ttr,ld.maas_ttr,ld.msttr,ld.mattr,ld.hdd,ld.mtld,ld.mtld_ma_wrap,ld.mtld_ma_bid]
        return list(map(lambda x: x(text), funcs))

#takes the tagged text as input and outputs simple counts
def quantity_features(tagged_tokens):
        pos_counts = pos_list
        for word, tag in tagged_tokens:
            pos_counts[tag] = pos_counts[tag] + 1
        num_terms, num_tokens = len(set(tagged_tokens)), len(tagged_tokens)
        counts = [value for key,value in sorted(pos_counts.items())]
        return [num_terms, num_tokens]+counts

def uncertainty_features(text):
        cls = Classifier()
        results = cls.predict(text)
        unc_ratio = sum([1 for r in results if r == 'U'])/len(results)
        return unc_ratio

#takes the raw text as input, and calculates polarity as the average polarity over the sentences
#for vader, takes the text split in sentences
def affect_features(text, sentences, n_sentences, blob):
        analyzer = SentimentIntensityAnalyzer()
        polarity_blob = sum([s.sentiment.polarity for s in blob.sentences]) / len(blob.sentences)
        polarity_vader = sum([analyzer.polarity_scores(s)['compound'] for s in sentences]) / n_sentences
        return polarity_blob,polarity_vader

def passiveness_features(sentences):
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

    count = 0
    for s in sentences:
        tagged = tagger.tag(nltk.word_tokenize(s))
        tags = map(lambda tup: tup[1], tagged)
        after_to_be = list(dropwhile(lambda tag: not tag.startswith("BE"), tags))
        nongerund = lambda tag: tag.startswith("V") and not tag.startswith("VBG")
        filtered = filter(nongerund, after_to_be)
        out = any(filtered)
        if bool(out): count+=1


    return(count/len(sentences))

