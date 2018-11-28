from model import *
from args import *

# This script train the TWEC model with the default parameter

args = parse_args()

# Train TWEC model

m = twec_model(args)

m.train_static()

m.train_temporal_embeddings()

#m.evaluate()