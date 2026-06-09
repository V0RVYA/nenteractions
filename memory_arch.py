import nengo
import nengo_spa as spa
import numpy as np
import helper as hp
from neural_dfa import DFA, InputVar, StateVar
from collections import UserDict

#dimentions and default vocabulary initialized here
d = 128
theta = 0.3
voc = spa.Vocabulary(d)


###################################
#     The Memory Architecture
###################################
# def ports_handler(ports, dim, vocab, theta, keys, nloc, vloc):
#     #The function below manages the signals to manipulate the ports using spa pointers for keys, tags and values
#     # -> need to split keys into nodes and vars -> handle that -> pass back to 2 distinct key dictionaries -> rejig all conditonals to work with it
#     ports = ports
#     keys = keys
#     nloc = nloc
#     vloc = vloc
#     theta = theta
#
#     def is_var_or_node(tag):
#         val = False
#         node = False
#         nodes_list = [vocab["T_CON"].v, vocab["T_DUP"].v, vocab["T_OPR"].v, vocab["T_SWI"].v]
#         if tag.dot(vocab.["T_VAR"].v) > theta:
#             val = True
#         elif:
#             for i in nodes_list:
#                 if tag.dot(i) > theta:
#                     node = True
#         return val, node
#
#
#     def process_port(t, x):
#         command = x[:dim]
#         temp1 = x[dim:dim*2]
#         temp2 = x[dim*2:dim*3]
#
#         if command.dot(vocab["P_LENGTH"].v) > theta:
#             pass
#         elif command.dot(vocab["P_ADD"].v) > theta:
#             print("cunt")
#             empty = hp.check_key_empty(keys, ports)
#             val_true, node_true = is_var_or_node(temp1)
#             if empty == 0:
#                 str_key = str(len(ports))
#                 vocab.populate(f"k_{str_key}")
#                 keys.append(vocab[f"k_{str_key}"].v)
#                 ports[vocab[f"k_{str_key}"].v] = temp1, temp2
#             elif empty != 0:
#                 ports[empty] = temp1, temp2
#             temp1 = np.zeros(dim)
#             temp2 = np.zeros(dim)
#         elif command.dot(vocab["P_COLLAPSE"].v) > theta:
#             b_key = np.argmax(vocab.dot[temp1])
#             key = vocab.keys[b_key]
#             ports.pop(key)
#             temp1 = np.zeros(dim)
#             temp2 = np.zeros(dim)
#         elif command.dot(vocab["P_TAG"].v) > theta:
#             b_key = np.argmax(vocab.dot[temp1])
#             key = vocab.keys[b_key]
#             temp1, temp2 = ports[key]
#             temp2 = np.zeros(dim)
#         elif command.dot(vocab["P_VALUE"].v) > theta:
#             b_key = np.argmax(vocab.dot[temp1])
#             key = vocab.keys[b_key]
#             temp1, temp2 = ports[key]
#             temp1 = np.zeros(dim)
#         elif command.dot(vocab["P_RULE"].v) > theta:
#             b_key1 = np.argmax(vocab.dot[temp1])
#             key1 = vocab.keys[b_key1]
#             b_key2 = np.argmax(vocab.dot[temp2])
#             key2 = vocab.keys[b_key2]
#             temp1, temp2 = ports[key1]
#             temp2, temp3 = ports[key2]
#         elif command.dot(vocab["P_ISNODE"].v) > theta:
#             pass
#         elif command.dot(vocab["P_ISVAR"].v) > theta:
#             pass
#         elif command.dot(vocab["P_PRIORITY"].v) > theta:
#             pass
#         return np.concatenate((temp1, temp2))
#     return process_port
#
class Ports(spa.Network):
    #takes the value of the port and the tag of type as arguments
    def __init__(self, vocab, theta, keys, nloc, vloc, label = "ports"):
        self.voc = vocab
        self.dim = vocab.dimensions
        self.theta = theta

        # dictionary that stores the ports 
        self.keys = keys
        self.nloc = nloc
        self.vloc = vloc
                 
        # self.inputz = nengo.Node(output = ports_handler(self.ports, self.dim, self.voc, self.theta, self.keys, self.nloc, self.vloc), size_in = 3*self.dim, size_out=2*self.dim, label = 'input')
        # self.outputz = spa.State(vocab=2*self.dim, label = 'output')
        # nengo.Connection(self.inputz, self.outputz.input)
        # self.tag2_out = spa.State(vocab=vocab, label = 'tag 2')
        # super().__init__() #this is if we want to alter initialization at the level of spa.Network defaults
    #######################################

# def pairs_handler(keys):
#     def process_pairs():
#         if command.dot()
#     return

class Pairs(spa.Network):
    def __init__(self, vocab, theta, keys, nloc, vloc, label = 'pairs'):
        self.voc = vocab
        self.theta = theta
        self.dim = vocab.dimensions 

        
class Redexes(spa.Network):
    # this network manages the active redexes
    def __init__(self, vocab, label = 'rbag'):
        self.vocab = vocab
        self.rbag_dict = {}
    pass

class Nodes(spa.Network):
    # this network manages the pairs and ports in GNet 
    def __init__(self, vocab, theta, pair, port, nodes, pairs, ports, keys, nloc, label = 'nodes'):
        super().__init__(label=label)
        self.vocab = vocab
        self.dim = vocab.dimensions
        self.theta = theta
        self.port = port
        self.pair = pair
        self.nodes_dict = nodes
        self.pairs = pairs
        self.ports = ports
        self.keys = keys
        self.nloc = nloc

        node_args = ["N_CREATE", "N_STORE", "N_LOAD", "N_EXCHANGE", "N_TAKE", "N_FREE", "N_NULL"]
        hp.add_voc(node_args, self.vocab)

        node_statevars = [("command", spa.SemanticPointer),
                          ("k_path", spa.SemanticPointer),
                          ("p_path", spa.SemanticPointer),
                          ("kcs_out", spa.SemanticPointer),
                          ("pcs_out", spa.SemanticPointer),
                          ("kl_out", spa.SemanticPointer),
                          ("ke_out", spa.SemanticPointer),
                          ("pe_out", spa.SemanticPointer),
                          ("kt_out", spa.SemanticPointer),
                          ("kf_out", spa.SemanticPointer)
                          ]
        
        node_table = {
                (voc["N_CREATE"], None, None): (voc["N_NULL"], InputVar("key", "kcs_out"), InputVar("pair", "pcs_out")),
                (voc["N_STORE"], None, None): (voc["N_NULL"], InputVar("key", "kcs_out"), InputVar("pair", "pcs_out")),
                (voc["N_LOAD"], None, None): (voc["N_NULL"], InputVar("key", "kl_out")),
                (voc["N_EXCHANGE"], None, None): (voc["N_NULL"], InputVar("key", "ke_out"), InputVar("pair", "pe_out")),
                (voc["N_TAKE"], None, None): (voc["N_NULL"], InputVar("key", "kt_out")),
                (voc["N_FREE"], None, None): (voc["N_NULL"], InputVar("key", "kf_out"))
                }
       
        # input to the Node system expects a 3*dim input [command, location_target(key), pair(key)]
        # command to statevar command
        # rest to following inputs to get passed along
        node_inputs = [
                ("key", self.dim),
                ("pair", self.dim),
                ]
        
        node_outputs = [("key_create", "kcs_out"),
                   ("pair_create", "pcs_out"),
                   ("key_ex","ke_out"),
                   ("pair_ex","pe_out"),
                   ("key_load","kl_out"),
                   ("key_take","kt_out"),
                   ("key_free","kf_out"),
                   ]
        # defining node functions
        def node_create(nodes_dict):
            #might need keys and nloc dictionary - unlikely tho
            def creator(t,x):
                key = x[:dim]
                pair = x[dim:2*dim]
                #nloc.append(key)
                nodes_dict[key] = pair
                return np.zeros(dim)
            return creator
        def node_load(nodes_dict):
            def loader(t,x):
                key = x
                return nodes_dict[key]
            return loader
        # connect exchanger to pairs => perform exchange
        def node_exchange(nodes_dict):
            def exchanger(t, x):
                key = x[:dim]
                pair = x[dim:2*dim]
                return_pair = nodes_dict[key]
                nodes_dict[key] = pair
                return return_pair
            return exchanger

        def node_take(nodes_dict):
            def taker(t,x):
                key = x
                return_pair = nodes_dict.pop(key)
                return return_pair
            return taker
        def node_free(nodes_dict):
            def freedom(t,x):
                key = x
                if nodes_dict[key]:
                    return 0
                else:
                    return 1

        with self:
            self.node_dfa = DFA(node_statevars, node_inputs, node_outputs, node_table, self.vocab, start=(voc["N_NULL"], None, None)) 

            # setting up inputs
            key_in = spa.State(self.vocab)
            pair_in = spa.State(self.vocab)
            nengo.Connection(key_in.output, self.node_dfa.input_key)
            nengo.Connection(pair_in.output, self.node_dfa.input_pair)

            
            

            #setting up outputs
            node_create = nengo.Node(output = node_create(self.nodes_dict), size_in = 2*self.dim, size_out = self.dim, label = 'node_create')
            node_load = nengo.Node(output = node_load(self.nodes_dict), size_in = self.dim, size_out = self.dim, label = 'node_load')
            node_exchange = nengo.Node(output = node_exchange(self.nodes_dict), size_in = 2*self.dim, size_out = self.dim, label = 'node_exchange')
            node_take = nengo.Node(output = node_take(self.nodes_dict), size_in = self.dim, size_out = self.dim, label = 'node_take')
            node_free = nengo.Node(output = node_free(self.nodes_dict), size_in = self.dim, size_out = 1, label = 'node_free')

            #node_outputs = [node_create, node_load, node_exchange, node_take, node_free]
            # print(node_dfa.ordered_outputs)
            # print(node_dfa.ordered_outputs[6])

            nengo.Connection(self.node_dfa.ordered_outputs[0], node_create[:self.dim]) 
            nengo.Connection(self.node_dfa.ordered_outputs[1], node_create[self.dim:2*self.dim])
            nengo.Connection(self.node_dfa.ordered_outputs[3], node_exchange[:self.dim])
            nengo.Connection(self.node_dfa.ordered_outputs[4], node_exchange[self.dim:2*self.dim])
            nengo.Connection(self.node_dfa.ordered_outputs[2], node_load)
            nengo.Connection(self.node_dfa.ordered_outputs[5], node_take)
            nengo.Connection(self.node_dfa.ordered_outputs[6], node_free)

  


        # output_states = [spa.state(self.vocab, label=outname) for outname, _ in outputs]

# class vars(spa.network):
#     # this network manages variables in gnet 
#     def __init__(self):
#         self.vars_dict = {}
    


class GNET(spa.Network):
    # this handles the representation of the entire graph network 
    def __init__(self, vocab, theta, label = 'net'):
        super().__init__(label=label)
        self.vocab = vocab
        self.dim = vocab.dimensions
        self.theta = theta

        self.nloc = []
        self.vloc = []
        self.keys = []
        self.ports = {}
        self.pairs = {}
        self.node_dict = {}
        # self.rbag_dict = {}
        self.vars_dict = {}
        #need to add function to handle g_enter
        gnet_args = ["G_NODE", "G_VAR", "G_NULL", "G_ENTER"]
        hp.add_voc(gnet_args, self.vocab)

        gnet_statevars = [("n_or_v", spa.SemanticPointer),
                          ("c_path", spa.SemanticPointer),
                          ("k_path", spa.SemanticPointer),
                          ("p_path", spa.SemanticPointer),
                          ("c_outn", spa.SemanticPointer),
                          ("k_outn", spa.SemanticPointer),
                          ("p_outn", spa.SemanticPointer),
                          ("c_outv", spa.SemanticPointer),
                          ("k_outv", spa.SemanticPointer),
                          ("p_outv", spa.SemanticPointer),
                          ]
        
        gnet_table = {
                (voc["G_NODE"], None, None): (voc["G_NULL"], InputVar("command", "c_outn"), InputVar("key", "k_outn"), InputVar("p_val","p_outn")),
                (voc["G_VAR"], None, None): (voc["G_NULL"], InputVar("command", "c_outv"), InputVar("key", "k_outv"), InputVar("p_val","p_outv"))
                }
       
        # input to the Gnet expects a 4*dim input [node/var, command, location_target(key), pair/port(key)]
        # node/var, to statevariable n_or_v
        # rest to following inputs to get passed along
        gnet_inputs = [
                ("command",self.dim),
                ("key", self.dim),
                ("p_val", self.dim),
                ]
        
        gnet_outputs = [("c_node", "c_outn"),
                   ("k_node", "k_outn"),
                   ("p_node", "p_outn"),
                   ("c_var", "c_outv"),
                   ("k_var", "k_outv"),
                   ("p_var", "p_outv"),
                   ]
        with self:
            self.gnet_dfa = DFA(gnet_statevars, gnet_inputs, gnet_outputs, gnet_table, self.vocab, start=(voc["G_NULL"], None, None)) 

            # setting up inputs
            command_in = spa.State(self.vocab)
            key_in = spa.State(self.vocab)
            pair_in = spa.State(self.vocab)
            nengo.Connection(command_in.output, self.gnet_dfa.input_command)
            nengo.Connection(key_in.output, self.gnet_dfa.input_key)
            nengo.Connection(pair_in.output, self.gnet_dfa.input_p_val)

            print (self.gnet_dfa.statevars.ordered_svs)


            self.pair = Pairs(self.vocab, self.theta, self.keys, self.nloc, self.vloc)
            self.port = Ports(self.vocab, self.theta, self.keys, self.nloc, self.vloc)
            self.node_manager = Nodes(self.vocab, self.theta, self.pair, self.port, self.node_dict, self.ports, self.pairs, self.keys, self.nloc)

            nengo.Connection(gnet_dfa.statevars.ordered_svs[4], node_manager.)
            print (self.node_manager.node_dfa.statevars.ordered_svs)



        # self.var = Vars()
        # self.rbag = Redexes(self.vocab, self.theta, self.pair, self.port, self.rbags, self.nodes, self.ports, self.pairs, self.keys, self.nloc)




# Adding port tags
tags = ["T_VAR","T_REF","T_ERA","T_NUM","T_CON","T_DUP","T_OPR","T_SWI", "TEST", "TRUE", "FALSE"]
hp.add_voc(tags, voc)

port_commands = ["P_ADD", "P_LENGTH", "P_NEW", "P_COLLAPSE", "P_TAG", "P_VALUE", "P_RULE", "P_ISNODE", "P_ISVAR", "P_PRIORITY"]
hp.add_voc(port_commands, voc)

# node_commands = ["N_FREE", "N_CREATE", "N_LOAD", "N_STORE", "N_EXCHANGE", "N_TAKE"]
# hp.add_voc(node_commands, voc)
# Adding the main interaction operators
interaction_rules = ["I_CALL","I_LINK","I_VOID","I_ERAS","I_COMM","I_ANNI","I_OPER","I_SWIT"]
hp.add_voc(interaction_rules, voc)
# Adding the numerical oprators 
# voc.populate("")

# Adding to the vocab the main tree node types
# voc.populate("F_ERA;F_REF;F_NUM;F_LCON;F_RCON;F_LDUP;F_RDUP;F_OPE;F_SWI;F_VAR")
with spa.Network() as model:
    gnet = GNET(voc, theta)



