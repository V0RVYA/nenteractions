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

def check_key_empty(keys, ports):
    for i in keys:
        e_key = i
        if i in ports:
            e_key = 0
        else:
            break
    return e_key


def from_vocab(pointer, vocab):
    similarity = np.dot(vocab.vectors, pointer)
    best_index = np.argmax(similarity)
    lis = list(vocab)
    word = lis[best_index]
    return word
    print (f"the value is: {k}")

