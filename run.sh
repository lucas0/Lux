#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

sudo docker stop $(sudo docker ps -a -q) > /dev/null
export BERT_BASE_DIR=~/Lux/res/bert/uncased_L-12_H-768_A-12

while true; do
    sed -r -i "s/^(random\.seed\()(.)(.*)$/\1$RANDOM\)/" $DIR/lux.py
    sudo -E python3 lux.py True
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
done
