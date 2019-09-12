# -*- coding: utf-8 -*-

"""Main module."""

# -*- coding: utf-8 -*-

"""Main module."""
from gensim.models.word2vec import Word2Vec, LineSentence, PathLineSentences
from gensim import utils
import os
import numpy as np
import glob
import logging
import copy


class TWEC:
    """
    Handles alignment between multiple slices of temporal text
    """
    def __init__(self, size=100, sg=0, siter=5, diter=5, ns=10, window=5, alpha=0.025,
                            min_count=5, workers=2, test = "test", opath="model", init_mode="hidden"):
        """

        :param size: Number of dimensions. Default is 100.
        :param sg: Neural architecture of Word2vec. Default is CBOW (). If 1, Skip-gram is employed.
        :param siter: Number of static iterations (epochs). Default is 5.
        :param diter: Number of dynamic iterations (epochs). Default is 5.
        :param ns: Number of negative sampling examples. Default is 10, min is 1.
        :param window: Size of the context window (left and right). Default is 5 (5 left + 5 right).
        :param alpha: Initial learning rate. Default is 0.025.
        :param min_count: Min frequency for words over the entire corpus. Default is 5.
        :param workers: Number of worker threads. Default is 2.
        :param test: Folder name of the diachronic corpus files for testing.
        :param opath: Name of the desired output folder. Default is model.
        :param init_mode: If \"hidden\" (default), initialize temporal models with hidden embeddings of the context;'
                            'if \"both\", initilize also the word embeddings;'
                            'if \"copy\", temporal models are initiliazed as a copy of the context model
                            (same vocabulary)
        """
        self.size = size
        self.sg =sg
        self.trained_slices = dict()
        self.gvocab = []
        self.static_iter = siter
        self.dynamic_iter =diter
        self.negative = ns
        self.window = window
        self.static_alpha = alpha
        self.dynamic_alpha = alpha
        self.min_count = min_count
        self.workers = workers
        self.test = test
        self.opath = opath
        self.init_mode = init_mode
        self.compass = None
        if not os.path.isdir(self.opath):
            os.makedirs(self.opath)
        with open(os.path.join(self.opath, "log.txt"), "w") as f_log:
            f_log.write(str("")) # todo args
            f_log.write('\n')
            logging.basicConfig(filename=os.path.realpath(f_log.name),
                                format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    def initialize_from_compass(self, model):
        print("Initializing temporal embeddings from the atemporal compass.")
        if self.init_mode == "copy":
            model = copy.deepcopy(self.compass)
        else:
            if self.compass.layer1_size != self.size:
                return Exception("Compass and Slice have different vector sizes")
            vocab_m = model.wv.index2word
            indices = [self.compass.wv.vocab[w].index for w in vocab_m]
            new_syn1neg = np.array([self.compass.syn1neg[index] for index in indices])
            model.syn1neg = new_syn1neg
            if self.init_mode == "both":
                new_syn0 = np.array([self.compass.wv.syn0[index] for index in indices])
                model.wv.syn0 = new_syn0
        model.learn_hidden = False
        model.alpha = self.dynamic_alpha
        model.iter = self.dynamic_iter
        return model

    def internal_trimming_rule(self, word, count, min_count):
        """
        Internal rule used to trim words
        :param word:
        :return:
        """
        if word in self.gvocab:
            return utils.RULE_KEEP
        else:
            return utils.RULE_DISCARD

    def train_model(self, sentences):
        model = None
        if self.compass == None or self.init_mode != "copy":
            model = Word2Vec(sg=self.sg, size=self.size, alpha=self.static_alpha, iter=self.static_iter,
                             negative=self.negative,
                             window=self.window, min_count=self.min_count, workers=self.workers)
            model.build_vocab(sentences, trim_rule=self.internal_trimming_rule if self.compass != None else None)
        if self.compass != None:
            model = self.initialize_from_compass(model)
        model.train(sentences, total_words=sum([len(s) for s in sentences]), epochs=model.iter, compute_loss=True)
        return model

    def train_compass(self, compass_text, overwrite=False):
        compass_exists = os.path.isfile(os.path.join(self.opath, "compass.model"))
        if compass_exists and overwrite is False:
            self.compass = Word2Vec.load(os.path.join(self.opath, "compass.model"))
            print("Compass loaded from file.")
        else:
            sentences = PathLineSentences(compass_text)
            sentences.input_files = [s for s in sentences.input_files if not os.path.basename(s).startswith('.')]
            print("Training the compass.")
            if compass_exists:
                print("Compass will be overwritten after training")
            self.compass = self.train_model(sentences)
            self.compass.save(os.path.join(self.opath, "compass.model"))

        self.gvocab = self.compass.wv.vocab

    def train_slice(self, slice_text, save=True):
        if self.compass == None:
            return Exception("Missing Compass")
        print("Training temporal embeddings: slice {}.".format(slice_text))

        sentences = LineSentence(slice_text)
        model = self.train_model(sentences)

        model_name = os.path.splitext(os.path.basename(slice_text))[0]

        self.trained_slices[model_name] = model

        if save:
            model.save(os.path.join(self.opath, model_name + ".model"))

        return self.trained_slices[model_name]

    def evaluate(self):
        mfiles = glob.glob(self.opath + '/*.model')
        mods = []
        vocab_len = -1
        for fn in sorted(mfiles):
            if "compass" in os.path.basename(fn): continue
            m = Word2Vec.load(fn)
            m.cbow_mean = True
            m.negative = self.negative
            m.window = self.window
            m.vector_size = self.size
            if vocab_len > 0 and vocab_len != len(m.wv.vocab):
                print(
                    "ERROR in evaluation: models with different vocab size {} != {}".format(vocab_len, len(m.wv.vocab)))
                return
            vocab_len = len(m.wv.vocab)
            mods.append(m)
        tfiles = glob.glob(self.test + '/*.txt')
        if len(mods) != len(tfiles):
            print(
                "ERROR in evaluation: number mismatch between the models ({}) in the folder {} and the test files ({}) in the folder {}".format(
                    len(mods), self.opath, len(tfiles), self.test))
            return
        mplps = []
        nlls = []
        for n_tfn, tfn in enumerate(sorted(tfiles)):
            sentences = LineSentence(tfn)
            # Taddy's code (see https://github.com/piskvorky/gensim/blob/develop/docs/notebooks/deepir.ipynb)
            llhd = np.array([m.score(sentences) for m in mods])  # (mods,sents)
            lhd = np.exp(llhd - llhd.max(axis=0))  # subtract row max to avoid numeric overload
            probs = (lhd / lhd.sum(axis=0)).mean(axis=1)  # (sents, mods)
            mplp = np.log(probs[n_tfn])
            mplps.append(mplp)

            nwords = len([w for s in sentences for w in s if w in mods[n_tfn].wv.vocab])
            nll = sum(llhd[n_tfn]) / (nwords)
            nlls.append(nll)
            print("Slice {} {}\n\t- Posterior log probability {:.4f}\n\tNormalized log likelihood {:.4f}".format(n_tfn,
                                                                                                                 tfn,
                                                                                                                 mplp,
                                                                                                                 nll))
        print("Mean posterior log probability: {:.4f}".format(sum(mplps) / (len(mplps))))
        print("Mean normalized log likelihood: {:.4f}".format(sum(nlls) / (len(nlls))))
