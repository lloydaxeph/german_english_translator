# German-English Translator using Multi-Head Self-Attention Transformer

## 1.0 About
This project aims to train and use a multi-head attention transformer model to translate **German** text to **English**.
Dataset that will be used in this implemenation is the [Multi30k dataset](https://github.com/multi30k/dataset).
The model that will be used is the [Multi-Head Self-Attention Transformer ](https://github.com/lloydaxeph/multi_head_attention_transformer) model I created from scratch.

## 2.0 Sample Implementation
### 2.1 Installation
Install the required packages.
```
pip3 install -r requirements.txt
```
### 2.2 Clone Model
Copy or clone the model from my repository to your project directory.
```
git clone https://github.com/lloydaxeph/multi_head_attention_transformer
```

### 2.3 Setup Dataset
Download the dataset as instructed in this [Multi30k dataset](https://github.com/multi30k/dataset) repository. Make sure the `dataset_path` variable in [config.py] is the path to your dataset similar to the following code:
```
dataset_path = 'multi30k-dataset/data/task1/raw/'
```

### 2.4 Model Training
To trigger training, simply input the following command in your terminal:
```
python3 train.py --epochs=100 --batch_size=64 --lr=0.001
```
Or you can just edit the parameters in variables in [config.py](https://github.com/lloydaxeph/imagenet_cnn_implementation/blob/master/config.py) and simply use:
```
python3 train.py
```

### 2.5 Model Testing
For testing, you can use the following command for testing where `--model_path` is the path of your pretrained model and `--num_samples` is the number of samples from your test dataset:
```
python3 test.py --model_path=mymodel.pt --num_samples=10
```
