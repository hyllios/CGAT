embedding_path="embeddings/matscholar-embedding.json"

# Training loop
for target in e_above_hull
do
  echo Training target "$target"
  train-CGAT --gpus 2 --target "$target" --fea-path "$embedding_path" --epochs 280 --clr-period 70 --data-path dcgat_1_000_features.pickle.gz --batch-size 2
done
