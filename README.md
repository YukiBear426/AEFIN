# Datasets
ETTh1, ETTh2, ETTm1, ETTm2, ExchangeRate, Weather, Electricity will be downloaded automatically


# install requirements

to run our code, **Please make sure your python version is >=3.8.**

pip install -r ./requirements.txt


# run scripts
source ./init.sh 


# run the model AEFIN
```python
# running AEFIN using (Dlinear Informer SCINet) backbone with output length 96, 168, 336, 720 on dataset (ExchangeRate Electricity ETTh1 ETTh2) with input window 96, and hyperparameter k
./scripts/run_fan_wandb.sh "Dlinear" "AEFIN" "ExchangeRate" "96 168 336 720" "cuda:0" 96 "{freq_topk:2}"

# running all baselines~
./scripts/run.sh "Informer" "No" "ExchangeRate" "96"  "cuda:0" 96 

# running all baselines~(DLinear backbone RevIN SAN DishTS) with output length 96, 168, 336, 720 on dataset ETTm1 ETTm2 with input window 96
./scripts/run.sh "Dlinear" "RevIN" "ExchangeRate" "96"  "cuda:0" 96

