import pandas as pd
import torch
import numpy as np
from gensim.models.word2vec import Word2Vec
from model_siamese import BatchProgramCC
from torch.autograd import Variable
import sys
from my_utilities import get_batch, eval_model_siamese, contrastive_loss
import argparse

parser = argparse.ArgumentParser(description='ASTNN with CL')
parser.add_argument('-t', '--train_on', type=str, help='Which dataset to use for training')
args = parser.parse_args()
train_on = args.train_on

margin         = 50
learning_rate  = 1e-3
lang           = "java"
HIDDEN_DIM     = 100
ENCODE_DIM     = 128
LABELS         = 1
EPOCHS         = 5
BATCH_SIZE     = 32
early_stopping = False
USE_GPU        = torch.cuda.is_available()

assert train_on in ["bcb", "scb"]

print("train on %s" % train_on)
print("margin = %d" % margin)
print("learning rate = %f" % learning_rate)

root                      = 'data/'
data_bcb_and_scb          = pd.read_pickle(root+lang+'/scb/blocks.pickle').sample(frac=1)
data_bcb_and_scb['label'] = 1 - data_bcb_and_scb['label']


word2vec = Word2Vec.load(root+lang+"/scb/embedding/node_w2v_128").wv
MAX_TOKENS = word2vec.vectors.shape[0]
EMBEDDING_DIM = word2vec.vectors.shape[1]
embeddings = np.zeros((MAX_TOKENS + 1, EMBEDDING_DIM), dtype="float32")
embeddings[:word2vec.vectors.shape[0]] = word2vec.vectors

# Initialize model
model = BatchProgramCC(EMBEDDING_DIM,HIDDEN_DIM,MAX_TOKENS+1,ENCODE_DIM,LABELS,BATCH_SIZE,
                            USE_GPU, embeddings)
if USE_GPU:
    model.cuda()

parameters = model.parameters()
#optimizer = torch.optim.Adamax(parameters, lr=learning_rate)
optimizer = torch.optim.Adam(parameters, lr=learning_rate)
loss_function = torch.nn.BCELoss()

# SCB has NaN in functionality_ID
if train_on == "bcb":
    data_train = data_bcb_and_scb[~data_bcb_and_scb['functionality_id'].isna()]
    
    data_test = data_bcb_and_scb[data_bcb_and_scb['functionality_id'].isna()]
elif train_on == "scb":
    data_train = data_bcb_and_scb[data_bcb_and_scb['functionality_id'].isna()]
    
    data_test = data_bcb_and_scb[~data_bcb_and_scb['functionality_id'].isna()]


for epoch in range(EPOCHS):
    print("Starting epoch %d" % epoch)
    sys.stdout.flush()
    i = 0
    while i < len(data_train):
        model.train()
        batch = get_batch(data_train, i, BATCH_SIZE)
        train1_inputs, train2_inputs, train_labels = batch
        if USE_GPU:
            train1_inputs, train2_inputs, train_labels = train1_inputs, train2_inputs, train_labels.cuda()
    
        model.zero_grad()
        model.batch_size = len(train_labels)
        model.hidden     = model.init_hidden()
    
        embeddings1 = model(train1_inputs)
        embeddings2 = model(train2_inputs)
    
        loss = contrastive_loss(embeddings1, embeddings2, Variable(train_labels), margin=margin)

        loss.backward()
        optimizer.step()
        i += BATCH_SIZE

    
    print("Starting evaluation after epoch %d" % epoch)
    sys.stdout.flush()
    f, similarity_scores = eval_model_siamese(model, data_test, margin, BATCH_SIZE, USE_GPU)
    sys.stdout.flush()

    if early_stopping and f<prev_epoch_f1:
        print("Lower F1 than previous epoch. Early stopping...")
        sys.stdout.flush()
        break
    else:
        prev_epoch_f1 = f