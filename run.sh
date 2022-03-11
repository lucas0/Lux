#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

docker stop $(docker ps -a -q) > /dev/null
export BERT_BASE_DIR=~/Lux/res/bert/uncased_L-12_H-768_A-12

while true; do
    rnd=$RANDOM
    echo $rnd
    sed -r -i "s/^(seed\s=\s)(.*)$/\1$rnd/" $DIR/lux.py
    for lr in 0.0001 0.0005 0.001; do
        for dp in 0.3 0.5 0.7; do
            for dim in 128 256 512; do
                python lux.py --lr $lr --dropout $dp --dense_dim $dim --regenerate_features 'emb' --env 'dev'
            done
        done
    done
    best_avg=$(sed -rn 's/^AVG:\s([0-9\.]*)$/\1/p' $DIR/data/bck_best/README.md)
    best_f1=$(sed -rn 's/^F1:\s([0-9\.]*)$/\1/p' $DIR/data/bck_best/README.md)
    c_avg=$(sed -rn '$ s/(.*)AVG:\s([0-9\.]*)(.*)$/\2/p' $DIR/results.txt)
    c_f1=$(sed -rn '$ s/(.*)F1:\s([0-9\.]*)(.*)$/\2/p' $DIR/results.txt)

    #compare current acc_avg and current f1 with the saved best
    h_avg=$(echo | awk -v x="$best_avg" -v y="$c_avg" '{if(x>=y) print "v1"; if(x<y) print "v2"}')
    h_f1=$(echo | awk -v x="$best_f1" -v y="$c_f1" '{if(x>=y) print "v1"; if(x<y) print "v2"}')

    if [ "$h_f1" = "v2" ]; then
        if [ "$h_avg" = "v2" ]; then
            echo "New best achieved! Saving..."
            bash $DIR/BCK/save_bck.sh
        fi
    fi
    find . -maxdepth 1 -name 'tmp*' -type d -exec rm -r {} +

    #cp feat_log.txt feat_log1.txt
    #cp log_test_folds.txt log_test_folds1.txt
    for lr in 0.0001 0.0005 0.001; do
        for dp in 0.3 0.5 0.7; do
            for dim in 128 256 512; do
                python lux.py --lr $lr --dropout $dp --dense_dim $dim --input_features 'only_bert' --env 'dev'
            done
        done
    done
    #also doing with only claims
    #python lux.py True only_claims
    #add the random seed to results.txt
    #sed -r -i "$ s/(.*)$/\1 ROOT: $rnd OnlyClaims/" $DIR/results.txt
    #exit 0
done
