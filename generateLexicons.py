import os,sys
import pandas

#weights assined to each level of subjectivity
strong_weight, weak_weight = 0.7, 0.3

cwd = os.path.abspath(os.path.dirname(sys.argv[0]))
res_path = cwd+"/res"

#MPQA subjective lexicon score creation
mpqa_path = res_path+"/subjectivity/MPQA/"
lex = pandas.read_csv(mpqa_path+"/subjclueslen1-HLTEMNLP05.tff", sep=' ')
lex = lex.drop(['len','stemmed1'],axis=1)
lex = lex.replace(to_replace=r'^word1=', value='', regex=True)
lex = lex.replace(to_replace=r'^type=strongsubj', value=strong_weight, regex=True)
lex = lex.replace(to_replace=r'^type=weaksubj', value=weak_weight, regex=True)

lex = lex.replace(to_replace=r'^pos1=adj', value='JJ', regex=True)
lex = lex.replace(to_replace=r'^pos1=noun', value='NN', regex=True)
lex = lex.replace(to_replace=r'^pos1=verb', value='VB', regex=True)
lex = lex.replace(to_replace=r'^pos1=adverb', value='RB', regex=True)
lex = lex.replace(to_replace=r'^pos1=anypos', value='ANY', regex=True)

lex = lex.replace(to_replace=r'^priorpolarity=(neutral|both)', value=0.0, regex=True)
lex = lex.replace(to_replace=r'^priorpolarity=negative', value=-1.0, regex=True)
lex = lex.replace(to_replace=r'^priorpolarity=positive', value=1.0, regex=True)

lex.to_csv(mpqa_path+"/lexicon.csv", index=False)
