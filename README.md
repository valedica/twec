# TWEC: Temporal Word Embeddings with a Compass (AAAI 2019)

This package contains Python code to build temporal word embeddings with a compass! 
One of the problems of temporal word embeddings is that they require alignment between corpora.  We propose a method to aligned distributional representation based on word2vec.  This method is efficient and it is based on a simple heuristic: we freeze one layer of the CBOW architecture and train temporal embedding on the other matrix. See the [paper](https://aaai.org/ojs/index.php/AAAI/article/view/4594) for more details.

## Reference

This work is based on the following paper ([link](https://aaai.org/ojs/index.php/AAAI/article/view/4594)). 

+ Di Carlo, V., Bianchi, F., & Palmonari, M. (2019). **Training Temporal Word Embeddings with a Compass**. Proceedings of the AAAI Conference on Artificial Intelligence, 33(01), 6326-6334. https://doi.org/10.1609/aaai.v33i01.33016326

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

