from gensim.models.word2vec import Word2Vec, LineSentence, PathLineSentences
from gensim import utils, matutils
from scipy.misc import logsumexp
import os
import numpy as np
import glob
import logging
import copy

gvocab = None
def my_rule(word, count, min_count):
    if word in gvocab:
        return utils.RULE_KEEP
    else:
        return utils.RULE_DISCARD

class twec_model(object):
    def __init__(self, args):
        self.args = args
        self.size = args.size
        self.sg = args.sg
        self.static_iter = args.siter
        self.dynamic_iter = args.diter
        self.negative = args.ns
        self.window = args.window
        self.static_alpha = args.alpha
        self.dynamic_alpha = args.alpha
        self.min_count = args.min_count
        self.workers = args.workers
        self.train = args.train
        self.test = args.test
        self.opath = args.opath
        self.init_mode = args.init_mode
        self.compass = None
        if not os.path.isdir(self.opath):
            os.makedirs(self.opath)
        with open(os.path.join(self.opath,  "log.txt"), "w") as f_log:
            f_log.write(str(self.args))
            f_log.write('\n')
            logging.basicConfig(filename=os.path.realpath(f_log.name),format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


    def initialize_from_compass(self, model):
        print("Initializing temporal embeddings from the atemporal compass.")
        if self.init_mode=="copy":
            model = copy.deepcopy(self.compass)
        else:
            vocab_m = model.wv.index2word
            indices = [self.compass.wv.vocab[w].index for w in vocab_m]
            new_syn1neg = np.array([self.compass.syn1neg[index]for index in indices])
            model.syn1neg = new_syn1neg
            if self.init_mode=="both":
                new_syn0 = np.array([self.compass.wv.syn0[index]for index in indices])
                model.wv.syn0 = new_syn0
        model.learn_hidden = False
        model.alpha = self.dynamic_alpha
        model.iter = self.dynamic_iter
        return model
    

    def train_model(self, sentences):
        model = None
        if self.compass == None or self.init_mode != "copy":
            model = Word2Vec(sg=self.sg, size=self.size, alpha=self.static_alpha, iter=self.static_iter, negative=self.negative,
                            window=self.window, min_count=self.min_count, workers=self.workers)
            model.build_vocab(sentences, trim_rule = my_rule if self.compass != None else None)
        if self.compass != None:
            model = self.initialize_from_compass(model)
        model.train(sentences, total_words=sum([len(s) for s in sentences]), epochs=model.iter, compute_loss=True)
        return model

    def train_static(self):
        if os.path.isfile(os.path.join(self.opath,"static.model")):
            self.compass = Word2Vec.load(os.path.join(self.opath,"static.model"))
            print("Stic model loaded.")
        else:
            sentences = PathLineSentences(self.train)
            sentences.input_files = [s for s in sentences.input_files if not os.path.basename(s).startswith('.')]
            print("Training static embeddings.")
            self.compass = self.train_model(sentences)
            self.compass.save(os.path.join(self.opath,"static.model"))
        global gvocab
        gvocab = self.compass.wv.vocab

    def train_temporal_embeddings(self):
        if self.compass == None:
                self.train_static()
        files = glob.glob(self.train+'/*.txt')
        tot_n_files = len(files)
        for n_file, fn in enumerate(sorted(files)):
            print("Training temporal embeddings: slice {} of {}.".format(n_file+1, tot_n_files))
            sentences = LineSentence(fn)
            model = self.train_model(sentences)
            model.save(os.path.join(self.opath,os.path.splitext(os.path.basename(fn))[0])+".model")

    def evaluate(self):
        mfiles = glob.glob(self.opath+'/*.model')
        mods = []
        vocab_len = -1
        for fn in sorted(mfiles):
            if "static" in os.path.basename(fn): continue
            m = Word2Vec.load(fn)
            m.cbow_mean = True
            m.negative = self.negative
            m.window = self.window
            m.vector_size = self.size
            if vocab_len > 0 and vocab_len != len(m.wv.vocab):
                print("ERROR in evaluation: models with different vocab size {} != {}".format(vocab_len, len(m.wv.vocab)))
                return
            vocab_len = len(m.wv.vocab)
            mods.append(m)
        tfiles = glob.glob(self.test+'/*.txt')
        if len(mods)!=len(tfiles):
            print("ERROR in evaluation: number mismatch between the models ({}) in the folder {} and the test files ({}) in the folder {}".format(len(mods),self.opath,len(tfiles),self.test))
            return
        mplps = []
        nlls = []
        for n_tfn, tfn in enumerate(sorted(tfiles)):
            sentences = LineSentence(tfn)
            # Taddy's code (see https://github.com/piskvorky/gensim/blob/develop/docs/notebooks/deepir.ipynb)
            llhd = np.array([ m.score(sentences) for m in mods]) # (mods,sents)
            lhd = np.exp(llhd - llhd.max(axis=0)) # subtract row max to avoid numeric overload
            probs =  (lhd/lhd.sum(axis=0)).mean(axis=1) # (sents, mods)
            mplp = np.log( probs[n_tfn] )
            mplps.append(mplp)

            nwords = len([w for s in sentences for w in s if w in mods[n_tfn].wv.vocab])
            nll = sum(llhd[n_tfn]) / (nwords)
            nlls.append(nll)
            print("Slice {} {}\n\t- Posterior log probability {:.4f}\n\tNormalized log likelihood {:.4f}".format(n_tfn,tfn,mplp,nll))
        print
        print("Mean posterior log probability: {:.4f}".format(sum(mplps)/(len(mplps))))
        print("Mean normalized log likelihood: {:.4f}".format(sum(nlls)/(len(nlls))))
