##########################################
#   Helper Functions for Nenteration
##########################################

import numpy as np


def vcos(u, v):
    mag = u.length * v.length
    dot = u.dot(v)
    if mag == 0: return dot
    return dot/mag

def add_voc(lis, vocab):
    for i, v in enumerate(lis):
        vocab.populate(v)

def check_key_empty(keys, ports):
    for i in keys:
        e_key = i
        if i in ports:
            e_key = 0
        else:
            break
    return e_key



