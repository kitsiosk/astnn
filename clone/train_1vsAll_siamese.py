import pandas as pd
import torch
import time
import numpy as np
import warnings
from gensim.models.word2vec import Word2Vec
from model_siamese import BatchProgramCC
from torch.autograd import Variable
from sklearn.metrics import precision_recall_fscore_support
warnings.filterwarnings('ignore')
import sys

margin = 50
lr = 1e-3
early_stopping = False
print("Margin=%d" % margin)
print("lr=%0.5f"% lr)

def get_batch(dataset, idx, bs):
    tmp = dataset.iloc[idx: idx+bs]
    x1, x2, labels = [], [], []
    for _, item in tmp.iterrows():
        x1.append(item['code_x'])
        x2.append(item['code_y'])
        labels.append([item['label']])
    return x1, x2, torch.FloatTensor(labels)

# Contrastive loss function
def contrastive_loss(embedding1, embedding2, label, margin=1.0):
    distance = torch.nn.functional.pairwise_distance(embedding1, embedding2)

    loss = torch.mean((1 - label) * 0.5 * torch.pow(distance, 2) +
                      label * 0.5 * torch.pow(torch.clamp(margin - distance, min=0.0), 2))
    # print(distance) # Should not vanish to zero
    return loss

def eval_model(model, test_data_t):
    model.eval()
    similarity_scores = []
    trues = []
    iTest = 0
    
    while iTest < len(test_data_t):
        batch = get_batch(test_data_t, iTest, BATCH_SIZE)
        iTest += BATCH_SIZE
        test1_inputs, test2_inputs, test_labels = batch
        if USE_GPU:
            test_labels = test_labels.cuda()
    
        model.batch_size = len(test_labels)
        model.hidden = model.init_hidden()
        with torch.no_grad():
            embeddings1 = model(test1_inputs)
            embeddings2 = model(test2_inputs)
        similarity_score = torch.nn.functional.cosine_similarity(embeddings1, embeddings2)
    
        loss = contrastive_loss(embeddings1, embeddings2, Variable(test_labels), margin=margin)
    
        # calc testing acc
        similarity_scores.extend(similarity_score.cpu())
        trues.extend(1 - test_labels.cpu().numpy())
    
    
    trues = np.array(trues)
    
    predicted_labels = np.array(similarity_scores) > 0.2
    p, r, f, _ = precision_recall_fscore_support(trues, predicted_labels, average='binary')
    acc = 1-np.sum(np.abs(predicted_labels-np.transpose(trues)))/trues.shape[0]
    print("F1=%.3f, P=%.3f, R=%.3f, A=%.3f for similarity threshold 0.2" % (f, p, r, acc))

    max_Acc = -np.inf
    for similarity_threshold_int in range(0, 10):
        similarity_threshold = similarity_threshold_int/10
        # Classify code pairs based on the similarity score and threshold
        predicted_labels = (np.array(similarity_scores) > similarity_threshold)
        acc = 1-np.sum(np.abs(predicted_labels-np.transpose(trues)))/trues.shape[0]
        P, R, F1, _ = precision_recall_fscore_support(predicted_labels, trues, average='binary', pos_label=1)
    
    
        if acc > max_Acc:
            max_Acc = acc
            best_similarity_threshold = similarity_threshold

    predicted_labels = np.array(similarity_scores) > best_similarity_threshold
    p, r, f, _ = precision_recall_fscore_support(trues, predicted_labels, average='binary')
    acc = 1-np.sum(np.abs(predicted_labels-np.transpose(trues)))/trues.shape[0]
    print("F1=%.3f, P=%.3f, R=%.3f, A=%.3f for similarity threshold %0.2f" % (f, p, r, acc, best_similarity_threshold))


    max_F1 = -np.inf
    for similarity_threshold_int in range(0, 10):
        similarity_threshold = similarity_threshold_int/10
        # Classify code pairs based on the similarity score and threshold
        predicted_labels = (np.array(similarity_scores) > similarity_threshold)
        acc = 1-np.sum(np.abs(predicted_labels-np.transpose(trues)))/trues.shape[0]
        P, R, F1, _ = precision_recall_fscore_support(predicted_labels, trues, average='binary', pos_label=1)
    
    
        if F1 > max_F1:
            max_F1 = F1
            best_similarity_threshold = similarity_threshold
            
    predicted_labels = np.array(similarity_scores) > best_similarity_threshold
    p, r, f, _ = precision_recall_fscore_support(trues, predicted_labels, average='binary')
    acc = 1-np.sum(np.abs(predicted_labels-np.transpose(trues)))/trues.shape[0]
    print("F1=%.3f, P=%.3f, R=%.3f, A=%.3f for similarity threshold %0.2f" % (f, p, r, acc, best_similarity_threshold))


    sys.stdout.flush()
    model.train()
    return f, similarity_scores

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
    all_data['label'] = 1 - all_data['label']

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
    print(USE_GPU)
    print('Start training...')
    sys.stdout.flush()
    
    all_functionalities = all_data['functionality_id'].unique()
    all_functionalities.sort()
    for ii in all_functionalities:
        # Initialize model
        model = BatchProgramCC(EMBEDDING_DIM,HIDDEN_DIM,MAX_TOKENS+1,ENCODE_DIM,LABELS,BATCH_SIZE,
                                    USE_GPU, embeddings)
        if USE_GPU:
            model.cuda()

        parameters = model.parameters()
        #optimizer = torch.optim.Adamax(parameters, lr=lr)
        optimizer = torch.optim.Adam(parameters, lr=lr)

        
        if lang == 'java':
            train_data_t = all_data[all_data['functionality_id'] != ii]
            test_data_t = all_data[all_data['functionality_id'] == ii]
            print()
            print("Starting %d. Size train: %d | Size test: %d" % (ii, len(train_data_t), len(test_data_t)))
            sys.stdout.flush()

        else:
            print("Not implemented yet")
            quit()

        # training procedure
        prev_epoch_f1 = 0
        for epoch in range(EPOCHS):
            model.train()
            print(epoch)
            sys.stdout.flush()

            start_time = time.time()
            # training epoch
            i = 0
            while i < len(train_data_t):
                batch = get_batch(train_data_t, i, BATCH_SIZE)
                train1_inputs, train2_inputs, train_labels = batch

                if USE_GPU:
                    train_labels = train_labels.cuda()

                model.zero_grad()
                model.batch_size = len(train_labels)
                model.hidden = model.init_hidden()

                embeddings1 = model(train1_inputs)
                embeddings2 = model(train2_inputs)

                loss = contrastive_loss(embeddings1, embeddings2, Variable(train_labels), margin=margin)
                loss.backward()
                optimizer.step()

                # if (i/BATCH_SIZE) % 100 == 0:
                #     print(i/BATCH_SIZE)
                #     print(loss.detach().cpu().numpy())
                #     t1 = torch.nn.functional.pairwise_distance(embeddings1, embeddings2).detach().cpu().numpy()
                #     t2 = train_labels.cpu()

                #     idx_clones = (t2 == 0).squeeze()
                #     idx_non_clones = (t2 == 1).squeeze()
                #     print("Clones:")
                #     #print(t1[idx_clones])
                #     print(t1[idx_clones].mean())
                #     print("Non clones:")
                #     #print(t1[idx_non_clones])
                #     print(t1[idx_non_clones].mean())

                #     f, similarity_scores = eval_model(model, test_data_t)
                #     print()
                i += BATCH_SIZE
            

            f, similarity_scores = eval_model(model, test_data_t)
            if early_stopping and f<=prev_epoch_f1:
                print("Lower F1 than previous epoch. Early stopping...")
                sys.stdout.flush()
                break
            else:
                sys.stdout.flush()
                prev_epoch_f1 = f
