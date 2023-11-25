import pandas as pd
import torch
import time
import numpy as np
import warnings
from gensim.models.word2vec import Word2Vec
from model import BatchProgramCC
from torch.autograd import Variable
from sklearn.metrics import precision_recall_fscore_support
warnings.filterwarnings('ignore')
import sys


def get_batch(dataset, idx, bs):
    tmp = dataset.iloc[idx: idx+bs]
    x1, x2, labels = [], [], []
    for _, item in tmp.iterrows():
        x1.append(item['code_x'])
        x2.append(item['code_y'])
        labels.append([item['label']])
    return x1, x2, torch.FloatTensor(labels)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Choose a dataset:[c|java]")
    parser.add_argument('--lang')
    args = parser.parse_args()
    if not args.lang:
        print("No specified dataset")
        exit(1)
    root = 'data/'
    lang = args.lang
    categories = [1]
    if lang == 'java':
        categories = [5]
    print("Train for ", str.upper(lang))
    sys.stdout.flush()
    all_data = pd.read_pickle(root+lang+'/all/blocks.pickle').sample(frac=1)
    #test_data = pd.read_pickle(root+lang+'/test/blocks.pkl').sample(frac=1)

    word2vec = Word2Vec.load(root+lang+"/all/embedding/node_w2v_128").wv
    MAX_TOKENS = word2vec.vectors.shape[0]
    EMBEDDING_DIM = word2vec.vectors.shape[1]
    embeddings = np.zeros((MAX_TOKENS + 1, EMBEDDING_DIM), dtype="float32")
    embeddings[:word2vec.vectors.shape[0]] = word2vec.vectors

    HIDDEN_DIM = 100
    ENCODE_DIM = 128
    LABELS = 1
    EPOCHS = 5
    BATCH_SIZE = 32
    USE_GPU = torch.cuda.is_available()

    model = BatchProgramCC(EMBEDDING_DIM,HIDDEN_DIM,MAX_TOKENS+1,ENCODE_DIM,LABELS,BATCH_SIZE,
                                   USE_GPU, embeddings)
    if USE_GPU:
        model.cuda()

    parameters = model.parameters()
    optimizer = torch.optim.Adamax(parameters)
    loss_function = torch.nn.BCELoss()

    #print(train_data)
    precision, recall, f1 = 0, 0, 0
    print('Start training...')
    sys.stdout.flush()
    #breakpoint()
    all_functionalities = all_data['functionality_id'].unique()
    all_functionalities.sort()
    for ii in all_functionalities:
        if lang == 'java':
            train_data_t = all_data[all_data['functionality_id'] != ii].sample(20)
            test_data_t = all_data[all_data['functionality_id'] == ii].sample(10)
            print()
            print("Starting %d. Size train: %d | Size test: %d" % (ii, len(train_data_t), len(test_data_t)))
            sys.stdout.flush()

        else:
            print("Not implemented yet")
            quit()

        # training procedure
        prev_epoch_f1 = 0
        for epoch in range(EPOCHS):
            print(epoch)
            sys.stdout.flush()

            start_time = time.time()
            # training epoch
            total_acc = 0.0
            total_loss = 0.0
            total = 0.0
            i = 0
            while i < len(train_data_t):
                batch = get_batch(train_data_t, i, BATCH_SIZE)
                i += BATCH_SIZE
                train1_inputs, train2_inputs, train_labels = batch
                if USE_GPU:
                    train1_inputs, train2_inputs, train_labels = train1_inputs, train2_inputs, train_labels.cuda()

                model.zero_grad()
                model.batch_size = len(train_labels)
                model.hidden = model.init_hidden()
                output = model(train1_inputs, train2_inputs)

                loss = loss_function(output, Variable(train_labels))
                loss.backward()
                optimizer.step()

            ###### Start testing
            predicts = []
            trues = []
            total_loss = 0.0
            total = 0.0
            i = 0
            while i < len(test_data_t):
                batch = get_batch(test_data_t, i, BATCH_SIZE)
                i += BATCH_SIZE
                test1_inputs, test2_inputs, test_labels = batch
                if USE_GPU:
                    test_labels = test_labels.cuda()

                model.batch_size = len(test_labels)
                model.hidden = model.init_hidden()
                output = model(test1_inputs, test2_inputs)

                loss = loss_function(output, Variable(test_labels))

                # calc testing acc
                predicted = (output.data > 0.5).cpu().numpy()
                predicts.extend(predicted)
                trues.extend(test_labels.cpu().numpy())
                total += len(test_labels)
                total_loss += loss.item() * len(test_labels)

            p, r, f, _ = precision_recall_fscore_support(trues, predicts, average='binary')
            print("Total testing results(P,R,F1):%.3f, %.3f, %.3f" % (p, r, f))
            sys.stdout.flush()

            if f<prev_epoch_f1:
                print("Lower F1 than prevous epoch. Early stopping...")
                sys.stdout.flush()
                break
            else:
                prev_epoch_f1 = f
