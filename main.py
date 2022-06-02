import argparse
from classification import Classification


parser = argparse.ArgumentParser()

parser.add_argument('-data',default='sentenceEmbedding.pickle', help='embedding pickle file')
parser.add_argument('-cls', help="svm / gru")
parser.add_argument('-lingf', help='[linguistic feature] TRUE / FALSE')

#parser.add_argument('-batch_size', default=128, type=int, help="batch size")
#parser.add_argument('-n_epoch', default=30, type=int, help="# of epoch")
#parser.add_argument('-learning_rate', default=1e-4, type=float, help="learning rate") #1e-5 for test
#parser.add_argument('-num_warmup', default=4000, type=int, help="# of warmup (about learning rate)") # 2000으로 test (transformer_lr1e-3_64.pt)
#parser.add_argument('-dropout', type=float,default=0.1, help="ratio of dropout")

opt = parser.parse_args()

#==========================================#
data = opt.data
classification = opt.cls
linguistic_feature = opt.lingf

#==========================================#

Classification(data, classification, linguistic_feature)
