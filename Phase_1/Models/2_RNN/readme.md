python ./inference.py --checkpoint  ./model/best20.pt \       
    --meta ./model/meta.json \        
    --input ../data/easy.tsv \
    --output output/easy_predictions.csv

EASY
Boundary position correctness (x1): 12 / 20
Grammar + POS correctness (x2): 1 / 20

MED
Boundary position correctness (x1): 14 / 20
Grammar + POS correctness (x2): 8 / 20

HARD
Boundary position correctness (x1): 16 / 20
Grammar + POS correctness (x2): 11 / 20

TEST
Boundary position correctness (x1): 19 / 30
Grammar + POS correctness (x2): 12 / 30