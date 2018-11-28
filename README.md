# twec: Temporal Word Embeddings with a Compass

This package contains Python code to build temporal word embeddings with a compass!

## Requirements

* Python 2.7+
* numpy
* Cython (optional)

You can use pip to install all the previous required packages (https://pypi.org/project/pip/). Cython is optional but very recommended (19x speed)!
```
pip install --upgrade numpy
pip install --upgrade cython
```
Twec relies on a slightly modiefied version of Gensim v.3.6.0 , provided in this package: do not move the *gensim* folder the project directory.

## Quickstart Instructions

1. Clone this repository to your computer;
2. Locate the directory *train* inside the twec directory and place there the input text splices of your diachronic corpus (e.g. "data/1990.txt", "data/1991.txt", ...)
3. Run the script *main.py* of twec. The output word embeddings will be placed inside the folder named *model*.
```
python main.py
```
If you want, you can change the default parameters. For more datails:
```
python main.py --help
```

