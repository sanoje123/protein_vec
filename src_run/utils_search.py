import numpy as np
import pandas as pd
import torch
from transformers import T5EncoderModel, T5Tokenizer
import re
import gc
import h5py
import torch
from collections import defaultdict
from torch import nn
from torch.utils.data import DataLoader
import faiss
import os



def load_database(lookup_database):
    #Build an indexed database
    d = lookup_database.shape[1]
    index = faiss.IndexFlatIP(d)    #IndexFlatIP is a type of FAISS index that uses inner product (equivalent to cosine similarity for normalized vectors) for measuring similarity. The argument d is the dimension of the vectors that the index will contain.
    faiss.normalize_L2(lookup_database) #normalizes the vectors in lookup_database using the L2 norm (Euclidean norm). This is done because cosine similarity, which is what IndexFlatIP uses, is a measure of similarity between two normalized vectors.
    index.add(lookup_database)

    return(index)


def query(index, queries, k=10):
    faiss.normalize_L2(queries) #normalizes the query vectors using the L2 norm (Euclidean norm). This is done because cosine similarity, which is what IndexFlatIP uses, is a measure of similarity between two normalized vectors.
    D, I = index.search(queries, k) #this line performs the actual search. For each query vector, it finds the k most similar vectors in the index. D is a 2D numpy array where each row contains the distances to the k nearest neighbors of the corresponding query vector. I is a 2D numpy array where each row contains the indices of the k nearest neighbors of the corresponding query vector.

    return(D, I)

def featurize_prottrans(sequences, model, tokenizer, device): 
    
    sequences = [(" ".join(sequences[i])) for i in range(len(sequences))]   #just addes spaces between characters in sequence
    sequences = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences]    #replace any occurance ofcharacters ‘U’, ‘Z’, ‘O’, and ‘B’ in the sequences with ‘X’
    ids = tokenizer.batch_encode_plus(sequences, add_special_tokens=True, padding=True) #used to convert these sequences into a format that can be fed into the model. This includes tokenization, adding special tokens (like [CLS] and [SEP] for BERT), and padding the sequences to the same length.
    input_ids = torch.tensor(ids['input_ids']).to(device)   #The input_ids and attention_mask are then converted into PyTorch tensors and moved to the specified device (which could be a GPU or CPU).
    attention_mask = torch.tensor(ids['attention_mask']).to(device)

    with torch.no_grad():   #torch.no_grad() context manager is used to disable gradient calculations during this step, as we’re just doing inference and not training the model.
        embedding = model(input_ids=input_ids, attention_mask=attention_mask)   #the model is then used to generate embeddings for these sequences

    embedding = embedding.last_hidden_state.cpu().numpy()   # the model’s output is extracted and moved to the CPU. This is a 3D tensor containing the hidden states of each token in each sequence.

    features = [] 
    for seq_num in range(len(embedding)):
        seq_len = (attention_mask[seq_num] == 1).sum()  #it calculates the actual length of the sequence (i.e., excluding padding) using the attention_mask
        seq_emd = embedding[seq_num][:seq_len-1]    #extracts the hidden states corresponding to the actual tokens in the sequence (excluding the last token)
        features.append(seq_emd)
    
    prottrans_embedding = torch.tensor(features[0])
    prottrans_embedding = torch.unsqueeze(prottrans_embedding, 0).to(device)
    
    return(prottrans_embedding)


def embed_vec(prottrans_embedding, model_deep, masks, device):
    padding = torch.zeros(prottrans_embedding.shape[0:2]).type(torch.BoolTensor).to(device)
    out_seq = model_deep.make_matrix(prottrans_embedding, padding)
    vec_embedding = model_deep(out_seq, masks)
    return(vec_embedding.cpu().detach().numpy())

def encode(sequences, model_deep, model, tokenizer, masks, device):
    i = 0
    embed_all_sequences=[]
    while i < len(sequences):
        protrans_sequence = featurize_prottrans(sequences[i:i+1], model, tokenizer, device)
        embedded_sequence = embed_vec(protrans_sequence, model_deep, masks, device)
        embed_all_sequences.append(embedded_sequence)
        i = i + 1
    return np.concatenate(embed_all_sequences, axis=0)