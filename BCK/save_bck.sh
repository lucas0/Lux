#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
LUX_DIR="$(dirname "$DIR")"

cp -r $LUX_DIR/data/folds $LUX_DIR/data/bck_best/
cp $LUX_DIR/data/data.csv $LUX_DIR/data/bck_best/
cp $LUX_DIR/data/hash.txt $LUX_DIR/data/bck_best/
cp $LUX_DIR/data/datasets/dataset.csv $LUX_DIR/data/bck_best/

v1=$(sed -rn 's/^seed\s=\s([0-9]*)$/ \1/p' $LUX_DIR/lux.py)
avg=$(sed -rn '$ s/(.*)AVG:\s([0-9\.]*)(.*)$/\2/p' $LUX_DIR/results.txt)
f1=$(sed -rn '$ s/(.*)F1:\s([0-9\.]*)(.*)$/\2/p' $LUX_DIR/results.txt)

#captures in group 1 the short sentence "seed on lux.py:"
#then substitutes the whole line for \1 plus the content of v1
#-i for sub in-place
#-r to use sed regex
sed -r -i "s/^(seed\son\slux\.py:)(.*)/\1$v1/" $LUX_DIR/data/bck_best/README.md
sed -r -i "s/^(AVG:\s)(.*)/\1$avg/" $LUX_DIR/data/bck_best/README.md
sed -r -i "s/^(F1:\s)(.*)/\1$f1/" $LUX_DIR/data/bck_best/README.md

echo "backed up data, hash and folds"
