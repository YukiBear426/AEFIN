# AEFIN-main
A novel non-stationary time series forecasting framework based on Fourier analysis and cross attention mechanism.

# Paper
🎉 **Congratulations!** Our paper **"Non-Stationary Time Series Forecasting Based on Fourier Analysis and Cross Attention Mechanism"** has been accepted at the <span style="color: #0066cc;">**[2025 International Joint Conference on Neural Networks (IJCNN)](https://2025.ijcnn.org/)**, Rome, Italy</span> ✨  

![image1](https://github.com/user-attachments/assets/cae7e806-1a26-4c32-abf8-0b5bba22ddfb)

The overall architecture of our model AEFIN. $X_t$ is the input data, the blue arrow indicates the direction of data flow, the blue dashed line indicates the direction of flow of the data used for residual connection, and the black arrow indicates the source of the data in the loss function.

# Datasets
The following datasets will be downloaded automatically:
- ETTh1
- ETTh2
- ETTm1
- ETTm2
- ExchangeRate
- Weather
- Electricity

# Install requirements
To run our code, **please make sure your Python version is >= 3.8.**

To install the required dependencies, run:

```bash
pip install -r ./requirements.txt
```

# Run scripts
To initialize the environment, run:

```bash
source ./init.sh
```

# Run our model AEFIN
To run our model AEFIN with the Dlinear Informer SCINet backbone and output lengths 96, 168, 336, and 720 on datasets such as ExchangeRate, Electricity, ETTh1, and ETTh2, use the following command:

```bash
./scripts/run_fan_wandb.sh "Dlinear" "AEFIN" "ExchangeRate" "96 168 336 720" "cuda:0" 96 "{freq_topk:2}"
```

# Run all baselines
To run all baseline models (without using any framework) on the dataset (such as ExchangeRate) with output lengths (such as 168) 、input lengths 96 and the backbone model (such as Informer), use:

```bash
./scripts/run.sh "Informer" "No" "ExchangeRate" "168" "cuda:0" 96
```
To run all baseline models on the dataset (such as ExchangeRate) with output lengths (such as 168)、input lengths 96 and the backbone model (such as Dlinear) combined the RevIN, SAN, and DishTS frameworks, use:

```bash
./scripts/run.sh "Dlinear" "RevIN" "ExchangeRate" "168" "cuda:0" 96
```


