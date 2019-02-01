# TWEC: Temporal Word Embeddings with a Compass

This package contains Python code to build temporal word embeddings with a compass!

## Reference

+ Di Carlo, V., Bianchi, F. & Palmonari, M. (2019, January). **Training Temporal Word Embeddings with a Compass**. AAAI 

## Quickstart Instructions

TWEC relies on a slightly modiefied version of Gensim v.3.6.0. 
To install the required packages, execute these commands in your virtualenv (tested with python 3.7.1):

```
git clone https://github.com/valedica/twec.git
pip install git+https://github.com/valedica/gensim.git
```
To start training your temporal word embeddings, just type:
```
python main.py
```
You can change the default parameters if you want. For more datails:
```
python main.py --help
```

