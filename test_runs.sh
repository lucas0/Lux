#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

docker stop $(docker ps -a -q) > /dev/null
export BERT_BASE_DIR=~/Lux/res/bert/uncased_L-12_H-768_A-12

DATA_DIR=~/Lux/data/datasets

rnd=19582
lr=0.0005
dp=0.5
dim=256

#echo $rnd
#sed -r -i "s/^(seed\s=\s)(.*)$/\1$rnd/" $DIR/lux.py

#SET DATASET TO V4+EM+T2
cp $DATA_DIR/bck_dataset_ablation.csv $DATA_DIR/dataset.csv
echo "V4+EM+T2: " >> $DIR/results.txt
python lux.py --lr $lr --dropout $dp --dense_dim $dim
python lux.py --lr $lr --dropout $dp --dense_dim $dim --input_features 'only_bert'

#ABLATION (Another script)
echo "GROUP ABLATION: " >> $DIR/results.txt
declare -a FEAT_LIST=("inf" "div" "qua" "aff" "sbj" "spe" "pau" "unc" "pas")
for rmv_feat in "${FEAT_LIST[@]}"; do
    echo "Results without $rmv_feat: " >> $DIR/results.txt
    python lux.py --regenerate_features "just_reload" --feat_list ${FEAT_LIST[@]/$rmv_feat} --lr $lr --dropout $dp --dense_dim $dim
done
