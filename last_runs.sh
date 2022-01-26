#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

docker stop $(docker ps -a -q) > /dev/null
export BERT_BASE_DIR=~/Lux/res/bert/uncased_L-12_H-768_A-12

DATA_DIR=~/Lux/data/datasets

rnd=
lr=
dp=
dim=

echo $rnd
sed -r -i "s/^(seed\s=\s)(.*)$/\1$rnd/" $DIR/lux.py

#SET DATASET TO V4
cp $DATA_DIR/veritas4.csv $DATA_DIR/dataset.csv
python lux.py --lr $lr --dropout $dp --dense_dim $dim
python lux.py --lr $lr --dropout $dp --dense_dim $dim --input_features 'only_bert'

#SET DATASET TO V4+EM+T2
cp $DATA_DIR/bck_dataset_ablation.csv $DATA_DIR/dataset.csv
python lux.py --lr $lr --dropout $dp --dense_dim $dim
python lux.py --lr $lr --dropout $dp --dense_dim $dim --input_features 'only_bert'

#DATASET TO V4+EM+T2# (only a different call in lux.py)
python lux.py --lr $lr --dropout $dp --dense_dim $dim --only_claims True
python lux.py --lr $lr --dropout $dp --dense_dim $dim --input_features 'only_bert' --only_claims True

#SET DATASET TO EM
cp $DATA_DIR/emergent_gold.csv $DATA_DIR/dataset.csv
python lux.py --lr $lr --dropout $dp --dense_dim $dim

#SET DATASET TO V4+T1
cp $DATA_DIR/bck_dataset_v4+t1.csv $DATA_DIR/dataset.csv
python lux.py --lr $lr --dropout $dp --dense_dim $dim

#SET DATASET TO V4+EM+T1.csv
cp $DATA_DIR/bck_dataset_v4+em+t1.csv $DATA_DIR/dataset.csv
python lux.py --lr $lr --dropout $dp --dense_dim $dim

#SET DATASET TO V4+EM
cp $DATA_DIR/bck_dataset_v4+em.csv $DATA_DIR/dataset.csv
python lux.py --lr $lr --dropout $dp --dense_dim $dim

#SET DATASET TO SNOPES
cp $DATA_DIR/snopes2019.csv $DATA_DIR/dataset.csv
python lux.py --lr $lr --dropout $dp --dense_dim $dim

#SET DATASET TO FEVER
cp $DATA_DIR/fever.csv $DATA_DIR/dataset.csv
python lux.py --lr $lr --dropout $dp --dense_dim $dim --only_claims True

#ABLATION (Another script)
declare -a FEAT_LIST=("inf" "div" "qua" "aff" "sbj" "spe" "pau" "unc" "pas")
for rmv_feat in "${FEAT_LIST[@]}"; do
    echo "Results without $rmv_feat: " >> $DIR/results.txt
    python lux.py --regenerate_features "just_reload" --feat_list ${FEAT_LIST[@]/$rmv_feat} --lr $lr --dropout $dp --dense_dim $dim
done

#ABLATION INDIVIDUAL!!!

