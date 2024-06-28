ENAMINE_DB=./results/enamine
ENAMINE_DATA=./data/processed/Enamine10k_scores.csv
CHEMBL203_DB=./results/chembl203
CHEMBL203_DATA=./data/processed/CHEMBL203.csv
export PYTHONPATH="$PYTHONPATH:.."


python experiments/run.py \
  --n-iter 5 --sample-size 100 --protein 4UNN \
  --embedding fingerprint molformer chemberta-mtr \
  --sampler greedy explore expected-improvement random --surrogate linear-empirical \
  --data $ENAMINE_DATA \
  --out $ENAMINE_DB
python scripts/analyze.py $ENAMINE_DB --out ./results/

python experiments/run.py \
  --n-iter 5 --sample-size 100 --protein CHEMBL203 \
  --embedding fingerprint molformer chemberta-mtr \
  --sampler greedy explore expected-improvement random --surrogate linear-empirical \
  --data $CHEMBL203_DATA \
  --out $CHEMBL203_DB
python scripts/analyze.py $CHEMBL203_DB --out ./results/
