val_data="$1/val"
test_data="$1/test"
embeddings="embeddings/matscholar-embedding.json"

train-CGAT --gpus 2 --target "$2" --data-path "$1" --val-path "$val_data" --test-path "$test_data" --fea-path "$embeddings" --epochs 390 --clr-period 70 --pretrained-model "$3" --only-residual
