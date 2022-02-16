#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

docker stop $(docker ps -a -q) > /dev/null
export BERT_BASE_DIR=~/Lux/res/bert/uncased_L-12_H-768_A-12

DATA_DIR=~/Lux/data/datasets

rnd=27966
lr=0.0001
dp=0.5
dim=256

echo $rnd
sed -r -i "s/^(seed\s=\s)(.*)$/\1$rnd/" $DIR/lux.py

##SET DATASET TO FEVER
#cp $DATA_DIR/fever.csv $DATA_DIR/dataset.csv
#echo "FEVER: " >> $DIR/results.txt
#python lux.py --lr $lr --dropout $dp --dense_dim $dim --only_claims True
#
#SET DATASET TO SNOPES
#cp $DATA_DIR/snopes2019.csv $DATA_DIR/dataset.csv
#echo "SNOPES: " >> $DIR/results.txt
#python lux.py --lr $lr --dropout $dp --dense_dim $dim

#SET DATASET TO V4
#cp $DATA_DIR/veritas4.csv $DATA_DIR/dataset.csv
#echo "V4: " >> $DIR/results.txt
#python lux.py --lr $lr --dropout $dp --dense_dim $dim
#python lux.py --lr $lr --dropout $dp --dense_dim $dim --input_features 'only_bert'

#SET DATASET TO EM
#cp $DATA_DIR/emergent_gold.csv $DATA_DIR/dataset.csv
#echo "EM: " >> $DIR/results.txt
#python lux.py --lr $lr --dropout $dp --dense_dim $dim

#SET DATASET TO V4+T1
#cp $DATA_DIR/bck_dataset_v4+t1.csv $DATA_DIR/dataset.csv
#echo "V4+T1: " >> $DIR/results.txt
#python lux.py --lr $lr --dropout $dp --dense_dim $dim

#SET DATASET TO V4+EM+T1.csv
#cp $DATA_DIR/bck_dataset_v4+em+t1.csv $DATA_DIR/dataset.csv
#echo "V4+EM+T1: " >> $DIR/results.txt
#python lux.py --lr $lr --dropout $dp --dense_dim $dim

#SET DATASET TO V4+EM
#cp $DATA_DIR/bck_dataset_v4+em.csv $DATA_DIR/dataset.csv
#echo "V4+EM: " >> $DIR/results.txt
#python lux.py --lr $lr --dropout $dp --dense_dim $dim

#DATASET TO V4+EM+T2# (only a different call in lux.py)
cp $DATA_DIR/bck_dataset_ablation.csv $DATA_DIR/dataset.csv
BCK/load_bck.sh
echo "V4+EM+T2#: " >> $DIR/results.txt
python lux.py --lr $lr --dropout $dp --dense_dim $dim --only_claims True --regenerate_features 'all'
python lux.py --lr $lr --dropout $dp --dense_dim $dim --input_features 'only_bert' --only_claims True

#SET DATASET TO V4+EM+T2
cp $DATA_DIR/bck_dataset_ablation.csv $DATA_DIR/dataset.csv
bash BCK/load_bck.sh
echo "V4+EM+T2: " >> $DIR/results.txt
python lux.py --lr $lr --dropout $dp --dense_dim $dim
python lux.py --lr $lr --dropout $dp --dense_dim $dim --input_features 'only_bert'

#ABLATION (Another script)
bash BCK/load_bck.sh
echo "GROUP ABLATION: " >> $DIR/results.txt
declare -a FEAT_LIST=("inf" "div" "qua" "aff" "sbj" "spe" "pau" "unc" "pas")
for rmv_feat in "${FEAT_LIST[@]}"; do
    echo "Results without $rmv_feat: " >> $DIR/results.txt
    python lux.py --regenerate_features "just_reload" --feat_list ${FEAT_LIST[@]/$rmv_feat} --lr $lr --dropout $dp --dense_dim $dim
done

#ABLATION INDIVIDUAL!!!
echo "INDIVIDUAL ABLATION: " >> $DIR/results.txt
NUM_FEATS=100
for ((i=0;i<=NUM_FEATS;i++)); do
        python lux.py --regenerate_features "just_reload" --lr $lr --dropout $dp --dense_dim $dim --drop_feat_idx $i
done

