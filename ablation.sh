#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

docker stop $(docker ps -a -q) > /dev/null
declare -a FEAT_LIST=("inf" "div" "qua" "aff" "sbj" "spe" "pau" "unc" "pas")

python lux.py
for rmv_feat in "${FEAT_LIST[@]}"; do
    #rnd=4419
    #sed -r -i "s/^(seed\s=\s)(.*)$/\1$rnd/" $DIR/lux.py
    echo "Results without $rmv_feat: " >> $DIR/results.txt
    python lux.py --regenerate_features "just_reload" --feat_list ${FEAT_LIST[@]/$rmv_feat} --env "deploy"

    #write which feat was removed on results.txt:
done
