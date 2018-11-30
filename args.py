import argparse

def parse_args():
        parser = argparse.ArgumentParser(description="Train temporal word embeddings with atemporal context")

        parser.add_argument('--size', type=int, default=100,
                            help='Number of dimensions. Default is 100.')

        parser.add_argument('--siter', type=int, default=5,
                            help='Number of static iterations (epochs). Default is 5.')

        parser.add_argument('--window', type=int, default=5,
                            help='Size of the context window (left and right). Default is 5 (5 left + 5 right).')

        parser.add_argument('--diter', type=int, default=5,
                            help='Number of dynamic iterations (epochs). Default is 5.')

        parser.add_argument('--sg', type=int, default=0,
                            help='Neural architecture of Word2vec. Default is CBOW (). If 1, Skip-gram is employed.')

        parser.add_argument('--ns', type=ns_type, default=10, 
                            help='Number of negative sampling examples. Default is 10, min is 1.')

        parser.add_argument('--alpha', type=float, default=0.025,
                            help='Initial learning rate. Default is 0.025.')

        parser.add_argument('--min_count', type=int, default=5,
                            help='Min frequency for words over the entire corpus. Default is 5.')

        parser.add_argument('--workers', type=int, default=2,
                            help='Number of worker threads. Default is 2.')

        parser.add_argument('--train', type=str, default='train',
                            help='Folder name of the diachronic corpus files for training.')

        parser.add_argument('--test', type=str, default='test',
                            help='Folder name of the diachronic corpus files for testing.')

        parser.add_argument('--opath', type=str, default='model',
                            help='Name of the desired output folder. Default is model.')

        parser.add_argument('--init_mode', default='hidden', choices=["hidden", "both", "copy"],
                            help='If \"hidden\" (default), initialize temporal models with hidden embeddings of the context;'+ 
                            'if \"both\", initilize also the word embeddings;'+
                            'if \"copy\", temporal models are initiliazed as a copy of the context model (same vocabulary)')

        args =  parser.parse_args()
        return args
        
def ns_type(x):
    x = int(x)
    if x < 1:
        raise argparse.ArgumentTypeError("Min negative sample number is 1.")
    return x
