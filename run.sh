sudo docker stop $(sudo docker ps -a -q) > /dev/null
BERT_BASE_DIR=~/Lux/res/bert/uncased_L-12_H-768_A-12
sudo -E python3 lux.py
