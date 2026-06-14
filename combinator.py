import nengo
import nengo_spa as spa
import numpy as np
import helper as hp

from neural_dfa import DFA, InputVar, StateVar
from memory_arch import GNET, Ports, Pairs, Redexes


#dimentions and default vocabulary initialized here
d = 128
theta = 0.3
voc = spa.Vocabulary(d)
voc.add("NULL", np.zeros(d))

class InteractionCombinator(spa.Network):
    def __init__(self, vocab, theta, keys, ports, nloc, vloc, label = "Interaction Combinator"):
        super().__init__(label = label)
    pass

class ERAS_Interaction(spa.Network):
    def __init__(self, vocab, theta, keys, ports, nloc, vloc, label = "ERAS Interaction"):
        super().__init__(label = label)
    pass

class LINK_Interaction(spa.Network):
    def __init__(self, vocab, theta, keys, ports, nloc, vloc, label = "LINK Interaction"):
        super().__init__(label = label)
    pass

#need user defined functions for this - lol no not right now
# class CALL_Interaction(spa.Network):
#     def __init__(self, vocab, theta, keys, ports, nloc, vloc, label = "CALL Interaction"):
#         super().__init__(label = label)
#     pass

class VOID_Interaction(spa.Network):
    def __init__(self, vocab, theta, keys, ports, nloc, vloc, label = "VOID Interaction"):
        super().__init__(label = label)
    pass

class COMM_Interaction(spa.Network):
    def __init__(self, vocab, theta, keys, ports, nloc, vloc, label = "COMM Interaction"):
        super().__init__(label = label)
    pass

# needs real number - coming soon to a computer near you
# class OPR_Interaction(spa.Network):
#     def __init__(self, vocab, theta, keys, ports, nloc, vloc, label = "OPR Interaction"):
#         super().__init__(label = label)
#     pass

class SWI_Interaction(spa.Network):
    def __init__(self, vocab, theta, keys, ports, nloc, vloc, label = "SWIT Interaction"):
        super().__init__(label = label)
    pass

with spa.Network() as model:

