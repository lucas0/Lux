#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

docker stop $(docker ps -a -q) > /dev/null
export BERT_BASE_DIR=~/Lux/res/bert/uncased_L-12_H-768_A-12

rnd=99999

echo $rnd
sed -r -i "s/^(seed\s=\s)(.*)$/\1$rnd/" $DIR/lux.py

BERT V4
LUX V4

LUX V4+EM+T2
BERT V4+EM+T2

BERT V4+EM+T2#
LUX V4+EM+T2#

LUX EM
LUX V4+T1
LUX V4+EM+T1
LUX V4+EM

LUX SNOPES
LUX FEVER

#ABLATION (Another script)

#ABLATION INDIVIDUAL!!!

python lux.py --lr $lr --dropout $dp --dense_dim $dim
python lux.py --lr $lr --dropout $dp --dense_dim $dim --input_features 'only_bert'
