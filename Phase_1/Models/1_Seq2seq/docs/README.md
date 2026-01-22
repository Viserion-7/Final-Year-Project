python inference.py --checkpoint model/best-epoch=05-val_em=0.5340.ckpt \
    --meta model/meta.json \
    --input data/user.tsv \
    --output output/predictions.csv

ToDo

✔ Implement Beam Search
✔ Add LR scheduler
✔ Try fine-tuning with augmented or domain-specific data
