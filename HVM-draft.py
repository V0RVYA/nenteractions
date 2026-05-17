import nengo
import nengo_spa as spa
import numpy as np

from neural_dfa import DFA, InputVar, StateVar
from collections import UserDict

#dimentions and default vocabulary initialized here
d = 256 
voc = spa.Vocabulary(d)

class Lexicon(UserDict):
    def __init__(self, mapping=None, *args, **kwargs):
        self.reverse = {}
        super().__init__(mapping, **kwargs)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.reverse[value] = key

    def update(self, other=(), *args, **kwds):
        super().update(other, **kwds)
        self.reverse.update(other=(((v, k) for k, v in other),))
        self.reverse.update(other=(((v, k) for k, v in kwds.items()),))

# class Ports(UserDict):
#     #takes the value of the port and the tag of type as arguments
#     def __init__(self):
#         self.ports = {}
#         super()._init_() #this is if we want to alter initialization at the level of UserDict defaults
#     with self:
#         def add_port(tag, value):
#
#             ports[value] = tag
#         def collapse_port(tag):
#             ports.pop(value)
#

# class Pairs (self, one, two):
#     #takes two ports as an argument and stores them as a tuple
#     pass
# # voc.populate
#
# class rbag(UserDict):
#     # this dictionary acts as a buffer to hold the current set of active redexes
#     def __init__(self, pair):
#         pass
#     pass
#
# class node(UserDict):
#     # this dictionary acts as a buffer to store all current nodes 
#     def __init__(self, pair):
#         pass
#     pass 
#
# class var(UserDict):
#     # this dictionary acts as a buffer that stores all the ports of the network 
#     def __init__(self, port):
#         pass
#     pass
#
# class GNET(UserDict):
#     # this dictionary stores the nodes, redexes, and variables of a single network 
#     def __init__(self, node, var, rbag):
#         pass
#     pass

# voc.populate("F_ERA;F_REF;F_NUM;F_LCON;F_RCON;F_LDUP;F_RDUP;F_OPE;F_SWI;F_VAR;P_MAIN;P_AUX;I_CALL;I_LINK;I_VOID;I_ERA;I_COMM;I_ANN;I_OP1;I_OP2;I_SW1;I_SW2")


#######################################
#
#               Vocab
#   Here we are adding the various 
#   parts of an interaction net 
#   combinator runtime to the vocab.
#
#######################################

# Adding port tags
voc.populate("P_MAIN;P_AUX;NULL;NEXT")

# Adding the main interaction operators
# voc.populate("I_CALL;I_LINK;I_VOID;I_ERA;I_COMM;I_ANN;I_OP1;I_OP2;I_SW1;I_SW2")

# Adding the numerical oprators 
# voc.populate("")

# Adding to the vocab the main tree node types
voc.populate("F_ERA;F_REF;F_NUM;F_LCON;F_RCON;F_LDUP;F_RDUP;F_OPE;F_SWI;F_VAR")


statevars_detector = [("tag1", spa.SemanticPointer),
                      ("tag2", spa.SemanticPointer),
                      ("pair", spa.SemanticPointer),
                      # ("val2", spa.SemanticPointer),
                      ("node", spa.SemanticPointer),
                      ("rbag", spa.SemanticPointer)]

# statevars_implementor = [("I_CALL", spa.SemanticPointer), 
#                          ("I_LINK", spa.SemanticPointer),
#                          ("statevar3", spa.SemanticPointer),
#                          ("statevar4", int),
#                          ("dummyin", spa.SemanticPointer),
#                          ("bananapass", spa.SemanticPointer)
#                          ]
#



outputs_detector = [("rbags", "rbag"),
                    ("nodes", "node")]
inputs = [
        ("a", d)
        ]

table_detector = {
        (voc["P_MAIN"], voc["P_MAIN"], None): (StateVar("pair","rbag")), 
        (voc["P_AUX"], None, None): (voc["NEXT"], StateVar("tag1","node")),
        (None, voc["P_AUX"], None): (voc["NEXT"], StateVar("tag1","node")),
        }

# table_implementor = {
#         (voc["F_VAR"], voc["F_VAR"], None, 1): (voc["I_LINK"], StateVar("?????", "???"), 0), 
#         (voc["Banana"], voc["Apple"], None, 0): (voc["Apple"], voc["Banana"], InputVar("a", "dummyin"), 1)
#         }
#

        
with spa.Network() as model:
    # kws = """Tree-Node Keywords:
    #     *                   F_ERA(ser)
    #     @                   F_REF(erence)
    #     <Numeric>           F_NUM(eric)
    #     (                   F_LCON(structor)
    #     )                   F_RCON(structor)
    #     {                   F_LDUP(licator)
    #     }                   F_RDUP(licator)
    #     $                   F_OPE(rator)
    #     ?                   F_SWI(tch)
    #     <alphanumeric>      F_VAR(iable)
    # """
    #
    detect_dfa = DFA(statevars_detector, inputs, outputs_detector, table_detector, voc, start=(voc["P_MAIN"], voc["P_MAIN"], None)) 
    # implement_dfa = DFA(statevars_implementor, )
    def direct_conf():
        conf = nengo.Config(nengo.Ensemble)
        conf[nengo.Ensemble].neuron_type = nengo.neurons.Direct()
        return conf

    with direct_conf(): 
        a = spa.State(voc)
        output_states = [spa.State(voc, label=outname) for outname, _ in outputs_detector[:-1]]
        # output_states.append(spa.State(len(dfa.output_nodes["rbags"]), subdimensions=1, label="strangefruit"))
    nengo.Connection(a.output, dfa.input_a) 


    for outnode, state in zip(dfa.ordered_outputs, output_states):
        nengo.Connection(outnode, state.input)
