DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
v1=$(sed -rn 's/^seed\son\s\lux\.py:\s([0-9]).*$/\1/p' $DIR/data/bck_best/README.md)

#captures in group 1 the short sentence "random.seed("
#then substitutes the whole line for \1 plus the content of v1 plus ')'
#-i for sub in-place
#-r to use sed regex
sed -r -i "s/^(random\.seed\()(.)(.*)$/\1$v1\)/" $DIR/lux.py

#copies the things back :)
cp -r $DIR/data/bck_best/folds $DIR/data/
cp $DIR/data/bck_best/data.csv $DIR/data/
cp $DIR/data/bck_best/hash.txt $DIR/data/
cp $DIR/data/bck_best/dataset.csv  $DIR/data/datasets/

echo "loaded folds, data and hash from bckp"

