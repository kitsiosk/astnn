import torch
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import sys

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
    #print(distance) # Should not vanish to zero
    return loss


def eval_model_baseline(model, test_data_t, batch_size, use_gpu):
    model.eval()
    similarity_scores = []
    trues = []
    iTest = 0
    
    while iTest < len(test_data_t):
        batch = get_batch(test_data_t, iTest, batch_size)
        iTest += batch_size
        test1_inputs, test2_inputs, test_labels = batch
        if use_gpu:
            test_labels = test_labels.cuda()
    
        model.batch_size = len(test_labels)
        model.hidden = model.init_hidden()
        with torch.no_grad():
            output = model(test1_inputs, test2_inputs)
        
        # calc testing acc
        similarity_scores.extend(output.data.cpu().numpy())
        trues.extend(test_labels.cpu().numpy())
    
    
    trues = np.array(trues)
    


    max_Acc = -np.inf
    for similarity_threshold_int in range(-5, 10):
        similarity_threshold = similarity_threshold_int/10
        # Classify code pairs based on the similarity score and threshold
        predicted_labels = (np.array(similarity_scores) > similarity_threshold)
        acc = 1-np.sum(np.abs(predicted_labels-trues))/trues.shape[0]
        P, R, F1, _ = precision_recall_fscore_support(predicted_labels, trues, average='binary', pos_label=1)
    
    
        if acc > max_Acc:
            max_Acc = acc
            best_similarity_threshold = similarity_threshold

    predicted_labels = np.array(similarity_scores) > best_similarity_threshold
    p, r, f, _ = precision_recall_fscore_support(trues, predicted_labels, average='binary')
    acc = 1-np.sum(np.abs(predicted_labels-trues))/trues.shape[0]
    print("F1=%.3f, P=%.3f, R=%.3f, A=%.3f for similarity threshold %0.2f" % (f, p, r, acc, best_similarity_threshold))


    max_F1 = -np.inf
    for similarity_threshold_int in range(-5, 10):
        similarity_threshold = similarity_threshold_int/10
        # Classify code pairs based on the similarity score and threshold
        predicted_labels = (np.array(similarity_scores) > similarity_threshold)
        acc = 1-np.sum(np.abs(predicted_labels-trueså))/trues.shape[0]
        P, R, F1, _ = precision_recall_fscore_support(predicted_labels, trues, average='binary', pos_label=1)
    
    
        if F1 > max_F1:
            max_F1 = F1
            best_similarity_threshold = similarity_threshold
            
    predicted_labels = np.array(similarity_scores) > best_similarity_threshold
    p, r, f, _ = precision_recall_fscore_support(trues, predicted_labels, average='binary')
    acc = 1-np.sum(np.abs(predicted_labels-trues))/trues.shape[0]
    print("F1=%.3f, P=%.3f, R=%.3f, A=%.3f for similarity threshold %0.2f" % (f, p, r, acc, best_similarity_threshold))


    sys.stdout.flush()
    model.train()
    return f, similarity_scores

def eval_model_siamese(model, test_data_t, margin, batch_size, use_gpu):
    model.eval()
    similarity_scores = []
    trues = []
    iTest = 0
    
    while iTest < len(test_data_t):
        batch = get_batch(test_data_t, iTest, batch_size)
        iTest += batch_size
        test1_inputs, test2_inputs, test_labels = batch
        if use_gpu:
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
    


    max_Acc = -np.inf
    for similarity_threshold_int in range(-5, 10):
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
    for similarity_threshold_int in range(-5, 10):
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