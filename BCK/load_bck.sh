DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
LUX_DIR="$(dirname "$DIR")"
v1=$(sed -rn 's/^seed\son\s\lux\.py:\s([0-9]*).*$/\1/p' $LUX_DIR/data/bck_best/README.md)

#captures in group 1 the short sentence "random.seed("
#then substitutes the whole line for \1 plus the content of v1 plus ')'
#-i for sub in-place
#-r to use sed regex
sed -r -i "s/^(seed\s=\s)([0-9]*).*$/\1$v1/" $LUX_DIR/lux.py

#copies the things back :)
cp -r $LUX_DIR/data/bck_best/folds $LUX_DIR/data/
cp $LUX_DIR/data/bck_best/data.csv $LUX_DIR/data/
cp $LUX_DIR/data/bck_best/hash.txt $LUX_DIR/data/
cp $LUX_DIR/data/bck_best/dataset.csv  $LUX_DIR/data/datasets/

echo "loaded folds, data and hash from bckp"

