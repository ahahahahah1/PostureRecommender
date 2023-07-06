
import argparse
import pickle

print("Inside parse_args")

parser = argparse.ArgumentParser(description='IN exercises')
parser.add_argument("-f", "--train_data", help="correct data", default= 'g1')
parser.add_argument("-l", "--learning_rate", help="learning rate", type=float, default= 0.0003)
parser.add_argument("-w", "--dtft_width", help="window size", type=int, default= 6)
parser.add_argument("-d", "--dropout", help="dropout prob", type=float, default= 0.5)
parser.add_argument("-ep", "--epoch", help="epoch", type=int, default= 100)
parser.add_argument("-sd", "--seed_val", help="seed", type=int, default= 112)

args = parser.parse_known_args()
args=args[0]
print(args)

with open('hyperparams_dictionary_copy.pkl', 'wb') as f:
    pickle.dump(args, f)
