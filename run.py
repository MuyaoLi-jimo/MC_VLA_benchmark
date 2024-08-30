import argparse
from vla_eval.file_interacting import get_models


def parse_args():
    parser = argparse.ArgumentParser()
    # Essential Args
    parser.add_argument('--bench', type=str, nargs='+', required=True)  #list
    parser.add_argument('--models', type=str, nargs='+', required=True) #list
    # Explicitly Set the Judge Model
    parser.add_argument('--judge', type=str, default=None)
    args = parser.parse_args()
    return args
    
def main():
    args = parse_args()
    get_models()
    
    
if __name__ == "__main__":
    main()
    