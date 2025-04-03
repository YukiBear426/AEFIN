# Attention-Enhanced Fourier-Integrated Neural Network For Non-stationary Time Series Forecasting.

🎉 **Congratulations!** Our paper **"Non-Stationary Time Series Forecasting Based on Fourier Analysis and Cross Attention Mechanism"** has been accepted at the <span style="color: #0066cc;">**[2025 International Joint Conference on Neural Networks (IJCNN)](https://2025.ijcnn.org/)**, Rome, Italy</span>. ✨  

The following is the primary architecture of our proposed model.

![image1](https://github.com/user-attachments/assets/57b024bf-94c4-4c6b-a948-a6250877ff57)

## Prepare datasets
ETTh1, ETTh2, ETTm1, ETTm2, ExchangeRate, Weather and Electricity will be downloaded automatically.
<br>
Illness and Traffic should be downloaded by yourself.

## Install requirements

**Please make sure your python version is >=3.8.**
```python
pip install -r ./requirements.txt
```

## Run scripts
```python
source ./init.sh
```

## Reproducing the baselines and our model
```python
# running AEFIN using (Dlinear Informer SCINet) backbone with output length 96, 168, 336, 720 on dataset (ExchangeRate Electricity ETTh1 ETTh2) with input window 96, and hyperparameter k
./scripts/run_fan_wandb.sh "Dlinear" "AEFIN" "ExchangeRate" "96 168 336 720" "cuda:0" 96 "{freq_topk:2}"

# running all baselines~
./scripts/run.sh "Informer" "No" "ExchangeRate" "96"  "cuda:0" 96 

# running all baselines~(DLinear backbone RevIN SAN DishTS) with output length 96, 168, 336, 720 on dataset ETTm1 ETTm2 with input window 96
./scripts/run.sh "Dlinear" "RevIN" "ExchangeRate" "96"  "cuda:0" 96
```
