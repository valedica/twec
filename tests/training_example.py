"""
Train embeddings using the TWEC model.
"""
import os
from twec.twec import TWEC


input_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
input_dir = os.path.join(input_dir, 'examples/training')
aligner = TWEC(
    size=50,
    sg=0,  # 1 = SKIP-GRAM; 0 = CBOW
    siter=20, 
    diter=20, 
    workers=os.cpu_count()
    )

if __name__ == "__main__": 
    aligner.train_compass(input_dir, overwrite=True) # COMPASS
    for file in os.listdir(input_dir):
        input_file = os.path.join(input_dir, file)
        aligner.train_slice(input_file, save=True) # SLICES