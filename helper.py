##########################################
#   Helper Functions for Nenteration
##########################################

import numpy as np
import nengo

def vcos(u, v):
    mag = u.length * v.length
    dot = u.dot(v)
    if mag == 0: return dot
    return dot/mag

def add_voc(lis, vocab):
    for i, v in enumerate(lis):
        vocab.populate(v)
        # reverse_voc[tuple(vocab[v].v)] = v

def check_key_empty(keys, mem_dict):
    if len(keys) == 0:
        return 0
    for v in keys:
        if v not in mem_dict:
            return v
    return 0


def from_vocab(pointer, vocab):
    similarity = np.dot(vocab.vectors, pointer)
    best_index = np.argmax(similarity)
    lis = list(vocab)
    word = lis[best_index]
    return word
    

