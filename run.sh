#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

docker stop $(docker ps -a -q) > /dev/null
export BERT_BASE_DIR=~/Lux/res/bert/uncased_L-12_H-768_A-12

while true; do
    rnd=$RANDOM
    #rnd=11191
    echo $rnd
    sed -r -i "s/^(random\.seed\()(.)(.*)$/\1$rnd\)/" $DIR/lux.py
    python lux.py --regenerate_features True
    #sudo -E python3 lux.py
    best_avg=$(sed -rn 's/^AVG:\s([0-9\.]*)$/\1/p' $DIR/data/bck_best/README.md)
    best_f1=$(sed -rn 's/^F1:\s([0-9\.]*)$/\1/p' $DIR/data/bck_best/README.md)
    c_avg=$(sed -rn '$ s/(.*)AVG:\s([0-9\.]*)(.*)$/\2/p' $DIR/results.txt)
    c_f1=$(sed -rn '$ s/(.*)F1:\s([0-9\.]*)(.*)$/\2/p' $DIR/results.txt)

    h_avg=$(echo | awk -v x="$best_avg" -v y="$c_avg" '{if(x>=y) print "v1"; if(x<y) print "v2"}')
    h_f1=$(echo | awk -v x="$best_f1" -v y="$c_f1" '{if(x>=y) print "v1"; if(x<y) print "v2"}')

    if [ "$h_avg" = "v2" ]; then
        if [ "$h_f1" = "v2" ]; then
            echo "New best achieved! Saving..."
            bash $DIR/BCK/save_bck.sh
        fi
    fi
    find . -maxdepth 1 -name 'tmp*' -type d -exec rm -r {} +

    #add the random seed to results.txt
    sed -r -i "$ s/(.*)$/\1 ROOT: $rnd/" $DIR/results.txt

    #also doing with only claims
    #python lux.py True only_claims
    #add the random seed to results.txt
    #sed -r -i "$ s/(.*)$/\1 ROOT: $rnd OnlyClaims/" $DIR/results.txt
done
