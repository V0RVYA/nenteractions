import nengo
import nengo_spa as spa
import numpy as np
import helper as hp
from neural_dfa import DFA, InputVar, StateVar
from collections import UserDict


###################################
#     The Memory Architecture
###################################
class Ports(spa.Network):
    #takes the value of the port and the tag of type as arguments
    def __init__(self, 
                 vocab:spa.Vocabulary, 
                 theta:float, 
                 keys:list[str], 
                 ports:dict[str, (np.ndarray, np.ndarray)],
                 nloc:list[spa.SemanticPointer], 
                 vloc:list[spa.SemanticPointer], 
                 label:str = "ports"):
        super().__init__(label = label)
        self.vocab = vocab
        self.dim = vocab.dimensions
        self.theta = theta

        # dictionary that stores the ports 
        self.keys = keys
        self.ports = ports
        self.nloc = nloc
        self.vloc = vloc
        
        port_args = ["P_NEW", "P_GTAG", "P_GVAL", "P_GRULE", "P_SWAP", "P_HIGH", "P_ADJUST", "P_NULL"]
        hp.add_voc(port_args, self.vocab)

        port_statevars = [("command", spa.SemanticPointer),
                          ("tk_path", spa.SemanticPointer),
                          ("vk_path", spa.SemanticPointer),
                          ("r_path", spa.SemanticPointer),
                          ("t_out", spa.SemanticPointer),
                          ("v_out",spa.SemanticPointer),
                          ("kgt_out", spa.SemanticPointer),
                          ("kgv_out", spa.SemanticPointer),
                          ("k1r_out", spa.SemanticPointer),
                          ("k2r_out", spa.SemanticPointer),
                          ("k1s_out", spa.SemanticPointer),
                          ("k2s_out", spa.SemanticPointer),
                          ("r_out", spa.SemanticPointer),
                          ("ka_out", spa.SemanticPointer),
                          ("nloc_out", spa.SemanticPointer)
                          ]
        # will need to add nloc out to P_adjust, once functions are made - know more about how implementor logic gonna work
        port_table = {
                (self.vocab["P_NEW"], None, None): (self.vocab["P_NULL"], InputVar("trk", "t_out"), InputVar("vk", "v_out")),
                (self.vocab["P_GTAG"], None, None): (self.vocab["P_NULL"], InputVar("trk", "kgt_out")),
                (self.vocab["P_GVAL"], None, None): (self.vocab["P_NULL"], InputVar("trk", "kgv_out")),
                (self.vocab["P_GRULE"], None, None): (self.vocab["P_NULL"], InputVar("trk", "k1r_out"), InputVar("vk", "k2r_out")),
                (self.vocab["P_SWAP"], None, None): (self.vocab["P_NULL"], InputVar("trk", "k2s_out"), InputVar("vk", "k2s_out")),
                (self.vocab["P_HIGH"], None, None): (self.vocab["P_NULL"], InputVar("trk", "r_out")),
                (self.vocab["P_ADJUST"], None, None): (self.vocab["P_NULL"], InputVar("trk", "ka_out"))
                }
       
        # input to the Node system expects a 3*dim input [command, location_target(key), pair(key)]
        # command to statevar command
        # rest to following inputs to get passed along
        port_inputs = [
                ("trk", self.dim),
                ("vk", self.dim),
                ]
        
        port_outputs = [("new_tag", "t_out"),
                        ("new_val", "v_out"),
                        ("tag_key","kgt_out"),
                        ("val_key","kgv_out"),
                        ("get_rule1","k1r_out"),
                        ("get_rule2","k2r_out"),
                        ("swap_key1","k1s_out"),
                        ("swap_key2","k2s_out"),
                        ("is_high","r_out"),
                        ("adjust_key","ka_out"),
                        ("adjust_loc","nloc_out")
                        ]

        interaction_rules = ["I_CALL","I_LINK","I_VOID","I_ERAS","I_COMM","I_ANNI","I_OPER","I_SWIT", "I_NULL"]
        hp.add_voc(interaction_rules, self.vocab)

        tags = ["T_VAR","T_REF","T_ERA","T_NUM","T_CON","T_DUP","T_OPR","T_SWI", "TEST", "TRUE", "FALSE"]
        hp.add_voc(tags, self.vocab)


        rule_statevars = [("key1", spa.SemanticPointer),
                          ("key2", spa.SemanticPointer),   
                          ("path", spa.SemanticPointer),
                          ("rule", spa.SemanticPointer)
                          ]
        rules_table = {
                (self.vocab["T_VAR"], self.vocab["T_VAR"], None): (self.vocab["I_LINK"], None, StateVar("key1","rule")),
                (self.vocab["T_VAR"], self.vocab["T_REF"], None): (self.vocab["I_LINK"], None, StateVar("key1","rule")),
                (self.vocab["T_VAR"], self.vocab["T_ERA"], None): (self.vocab["I_LINK"], None, StateVar("key1","rule")),
                (self.vocab["T_VAR"], self.vocab["T_NUM"], None): (self.vocab["I_LINK"], None, StateVar("key1","rule")),
                (self.vocab["T_VAR"], self.vocab["T_CON"], None): (self.vocab["I_LINK"], None, StateVar("key1","rule")),
                (self.vocab["T_VAR"], self.vocab["T_DUP"], None): (self.vocab["I_LINK"], None, StateVar("key1","rule")),
                (self.vocab["T_VAR"], self.vocab["T_OPR"], None): (self.vocab["I_LINK"], None, StateVar("key1","rule")),
                (self.vocab["T_VAR"], self.vocab["T_SWI"], None): (self.vocab["I_LINK"], None, StateVar("key1","rule")),
                (self.vocab["T_REF"], self.vocab["T_VAR"], None): (self.vocab["I_LINK"], None, StateVar("key1","rule")),
                (self.vocab["T_REF"], self.vocab["T_REF"], None): (self.vocab["I_VOID"], None, StateVar("key1","rule")),
                (self.vocab["T_REF"], self.vocab["T_ERA"], None): (self.vocab["I_VOID"], None, StateVar("key1","rule")),
                (self.vocab["T_REF"], self.vocab["T_NUM"], None): (self.vocab["I_VOID"], None, StateVar("key1","rule")),
                (self.vocab["T_REF"], self.vocab["T_CON"], None): (self.vocab["I_CALL"], None, StateVar("key1","rule")),
                (self.vocab["T_REF"], self.vocab["T_DUP"], None): (self.vocab["I_ERAS"], None, StateVar("key1","rule")),
                (self.vocab["T_REF"], self.vocab["T_OPR"], None): (self.vocab["I_CALL"], None, StateVar("key1","rule")),
                (self.vocab["T_REF"], self.vocab["T_SWI"], None): (self.vocab["I_CALL"], None, StateVar("key1","rule")),
                (self.vocab["T_ERA"], self.vocab["T_VAR"], None): (self.vocab["I_LINK"], None, StateVar("key1","rule")),
                (self.vocab["T_ERA"], self.vocab["T_REF"], None): (self.vocab["I_VOID"], None, StateVar("key1","rule")),
                (self.vocab["T_ERA"], self.vocab["T_ERA"], None): (self.vocab["I_VOID"], None, StateVar("key1","rule")),
                (self.vocab["T_ERA"], self.vocab["T_NUM"], None): (self.vocab["I_VOID"], None, StateVar("key1","rule")),
                (self.vocab["T_ERA"], self.vocab["T_CON"], None): (self.vocab["I_ERAS"], None, StateVar("key1","rule")),
                (self.vocab["T_ERA"], self.vocab["T_DUP"], None): (self.vocab["I_ERAS"], None, StateVar("key1","rule")),
                (self.vocab["T_ERA"], self.vocab["T_OPR"], None): (self.vocab["I_ERAS"], None, StateVar("key1","rule")),
                (self.vocab["T_ERA"], self.vocab["T_SWI"], None): (self.vocab["I_ERAS"], None, StateVar("key1","rule")),
                (self.vocab["T_NUM"], self.vocab["T_VAR"], None): (self.vocab["I_LINK"], None, StateVar("key1","rule")),
                (self.vocab["T_NUM"], self.vocab["T_REF"], None): (self.vocab["I_VOID"], None, StateVar("key1","rule")),
                (self.vocab["T_NUM"], self.vocab["T_ERA"], None): (self.vocab["I_VOID"], None, StateVar("key1","rule")),
                (self.vocab["T_NUM"], self.vocab["T_NUM"], None): (self.vocab["I_VOID"], None, StateVar("key1","rule")),
                (self.vocab["T_NUM"], self.vocab["T_CON"], None): (self.vocab["I_ERAS"], None, StateVar("key1","rule")),
                (self.vocab["T_NUM"], self.vocab["T_DUP"], None): (self.vocab["I_ERAS"], None, StateVar("key1","rule")),
                (self.vocab["T_NUM"], self.vocab["T_OPR"], None): (self.vocab["I_OPER"], None, StateVar("key1","rule")),
                (self.vocab["T_NUM"], self.vocab["T_SWI"], None): (self.vocab["I_SWIT"], None, StateVar("key1","rule")),
                (self.vocab["T_CON"], self.vocab["T_VAR"], None): (self.vocab["I_LINK"], None, StateVar("key1","rule")),
                (self.vocab["T_CON"], self.vocab["T_REF"], None): (self.vocab["I_CALL"], None, StateVar("key1","rule")),
                (self.vocab["T_CON"], self.vocab["T_ERA"], None): (self.vocab["I_ERAS"], None, StateVar("key1","rule")),
                (self.vocab["T_CON"], self.vocab["T_NUM"], None): (self.vocab["I_ERAS"], None, StateVar("key1","rule")),
                (self.vocab["T_CON"], self.vocab["T_CON"], None): (self.vocab["I_ANNI"], None, StateVar("key1","rule")),
                (self.vocab["T_CON"], self.vocab["T_DUP"], None): (self.vocab["I_COMM"], None, StateVar("key1","rule")),
                (self.vocab["T_CON"], self.vocab["T_OPR"], None): (self.vocab["I_COMM"], None, StateVar("key1","rule")),
                (self.vocab["T_CON"], self.vocab["T_SWI"], None): (self.vocab["I_COMM"], None, StateVar("key1","rule")),
                (self.vocab["T_DUP"], self.vocab["T_VAR"], None): (self.vocab["I_LINK"], None, StateVar("key1","rule")),
                (self.vocab["T_DUP"], self.vocab["T_REF"], None): (self.vocab["I_ERAS"], None, StateVar("key1","rule")),
                (self.vocab["T_DUP"], self.vocab["T_ERA"], None): (self.vocab["I_ERAS"], None, StateVar("key1","rule")),
                (self.vocab["T_DUP"], self.vocab["T_NUM"], None): (self.vocab["I_ERAS"], None, StateVar("key1","rule")),
                (self.vocab["T_DUP"], self.vocab["T_CON"], None): (self.vocab["I_COMM"], None, StateVar("key1","rule")),
                (self.vocab["T_DUP"], self.vocab["T_DUP"], None): (self.vocab["I_ANNI"], None, StateVar("key1","rule")),
                (self.vocab["T_DUP"], self.vocab["T_OPR"], None): (self.vocab["I_COMM"], None, StateVar("key1","rule")),
                (self.vocab["T_DUP"], self.vocab["T_SWI"], None): (self.vocab["I_COMM"], None, StateVar("key1","rule")),
                (self.vocab["T_OPR"], self.vocab["T_VAR"], None): (self.vocab["I_LINK"], None, StateVar("key1","rule")),
                (self.vocab["T_OPR"], self.vocab["T_REF"], None): (self.vocab["I_CALL"], None, StateVar("key1","rule")),
                (self.vocab["T_OPR"], self.vocab["T_ERA"], None): (self.vocab["I_ERAS"], None, StateVar("key1","rule")),
                (self.vocab["T_OPR"], self.vocab["T_NUM"], None): (self.vocab["I_OPER"], None, StateVar("key1","rule")),
                (self.vocab["T_OPR"], self.vocab["T_CON"], None): (self.vocab["I_COMM"], None, StateVar("key1","rule")),
                (self.vocab["T_OPR"], self.vocab["T_DUP"], None): (self.vocab["I_COMM"], None, StateVar("key1","rule")),
                (self.vocab["T_OPR"], self.vocab["T_OPR"], None): (self.vocab["I_ANNI"], None, StateVar("key1","rule")),
                (self.vocab["T_OPR"], self.vocab["T_SWI"], None): (self.vocab["I_COMM"], None, StateVar("key1","rule")),
                (self.vocab["T_SWI"], self.vocab["T_VAR"], None): (self.vocab["I_LINK"], None, StateVar("key1","rule")),
                (self.vocab["T_SWI"], self.vocab["T_REF"], None): (self.vocab["I_CALL"], None, StateVar("key1","rule")),
                (self.vocab["T_SWI"], self.vocab["T_ERA"], None): (self.vocab["I_ERAS"], None, StateVar("key1","rule")),
                (self.vocab["T_SWI"], self.vocab["T_NUM"], None): (self.vocab["I_SWIT"], None, StateVar("key1","rule")),
                (self.vocab["T_SWI"], self.vocab["T_CON"], None): (self.vocab["I_COMM"], None, StateVar("key1","rule")),
                (self.vocab["T_SWI"], self.vocab["T_DUP"], None): (self.vocab["I_COMM"], None, StateVar("key1","rule")),
                (self.vocab["T_SWI"], self.vocab["T_OPR"], None): (self.vocab["I_COMM"], None, StateVar("key1","rule")),
                (self.vocab["T_SWI"], self.vocab["T_SWI"], None): (self.vocab["I_ANNI"], None, StateVar("key1","rule"))
                }

        rules_inputs = [
                ("dummyin", self.dim),
                ]
        rules_outputs = [
                ("rule_out","rule")
                ]
        # want:
        # A new entry in the port dictionary containing the tag + value
        # to return the key to the tag/value in the port dictionary
        def new_port(keys, ports):
            def new(t,x):
                tag = x[:self.dim]
                value = x[self.dim:2*self.dim]
                empty = hp.check_key_empty(keys, ports)
                if empty == 0:
                    str_key = str(len(ports))
                    vocab.populate(f"K_{str_key}")
                    keys.append(f"K_{str_key}")
                    ports[f"K_{str_key}"] = tag, value
                    print(ports)
                    return vocab[f"K_{str_key}"].v
                elif empty != 0:
                    name = hp.from_vocab(empty, self.vocab)
                    ports[name] = tag, value
                    print(ports)
                    return empty
            return new
        # given the key, want the tag array passed along
        def port_tag(keys, ports):
            def get_tag(t,x):
                key = x
                key_name = hp.from_vocab(key, self.vocab)
                if key_name in keys and key_name in list(ports.keys()):
                    tag, val = ports[key_name]
                    return tag
                else:
                    return np.zeros(self.dim)
            return get_tag
        # given the key, want the value array passed along
        def port_val(keys, ports):
            def get_val(t,x):
                key = x
                key_name = hp.from_vocab(key, self.vocab)
                if key_name in keys and key_name in list(ports.keys()):
                    tag, val = ports[key_name]
                    return val
                else:
                    return np.zeros(self.dim)
            return get_val
            
        #remove -> inside of adjust port -> ignore this
        def port_is_node(keys, ports):
            def is_node(t,x):
                key = x
                key_name = hp.from_vocab(key, self.vocab)
                nodes_types = ["T_CON", "T_DUP", "T_OPR", "T_SWI"]
                if ports[key_name]:
                    if key_name in nodes_types:
                        return 1
                else:
                    return 0
            return is_node
        
        #remove -> inside of adjust port -> ignore this
        def port_is_var(keys, ports):
            def is_var(t,x):
                key = x
                key_name = hp.from_vocab(key, self.vocab)
                if ports[key_name]:
                    if key_name == "T_VAR":
                        return 1
                else:
                    return 0
            return is_var
        # given two keys, want to compare the two tags 
        #if the tag for key 2, has a higher index in ordered_tags than the tag for key 1 return 1, else return 0
        # this one is definitely broken -> not actually pulling out the tags
        def port_swap(keys, ports):
            ordered_tags = ["T_VAR","T_REF","T_ERA","T_NUM","T_CON","T_DUP","T_OPR","T_SWI"]
            def should_swap(t,x):
                nonlocal ordered_tags
                key1 = x[:self.dim]
                key2 = x[self.dim:2*self.dim]
                # if both keys are in ordered tags, and if b > a, return 1
                if (key1 @ key1) >= self.theta and (key2 @ key2) >= self.theta:
                    tag1 = hp.from_vocab(key1, self.vocab)
                    tag2 = hp.from_vocab(key2, self.vocab)
                    a = ordered_tags.index(tag1)
                    b = ordered_tags.index(tag2)
                    if b > a:
                        return 1
                return 0
            return should_swap
        # here given a rule, if rule is in list high_rule, return 1, else return 0
        def high_rule():
            def is_high(t,x):
                rule = x
                rule_name = hp.from_vocab(rule, self.vocab)
                high_rules = ["I_LINK", "I_VOID", "I_ERAS", "I_ANNI"]
                if rule_name in high_rules:
                    return 1
                else:
                    return 0
            return is_high
        #ignore this for now
        def adjust_port(keys, ports, nloc, vloc):
            def adjust(t,x):
                # remove is_var and is_node -> logic only used here
                # need to figure out how to perform the logic, of adjusting the pair value to the location of the stored value in nodes/vars
                # replace value with key pointing to value stored somewhere in nodes/vars
                #will wait til functions made to implement
                pass
            return adjust

        with self:
            self.port_dfa = DFA(port_statevars, port_inputs, port_outputs, port_table, self.vocab, start=(self.vocab["P_NULL"], self.vocab["P_NULL"], None)) 
            self.rule_dfa = DFA(rule_statevars, rules_inputs, rules_outputs, rules_table, self.vocab, start=(self.vocab["I_NULL"], self.vocab["I_NULL"], None)) 
            # setting up inputs
            self.trk_in = spa.State(self.vocab, label = 'tag,rule,key in')
            self.vk_in = spa.State(self.vocab, label = 'value,key in')
            self.dummyin = spa.State(self.vocab, label = 'dummy in')

            nengo.Connection(self.trk_in.output, self.port_dfa.input_trk)
            nengo.Connection(self.vk_in.output, self.port_dfa.input_vk)
            nengo.Connection(self.dummyin.output, self.rule_dfa.input_dummyin)

            #setting up outputs
            self.port_new = nengo.Node(output = new_port(self.keys, self.ports), size_in = 2*self.dim, size_out = self.dim, label = 'port_new')
            self.port_tag = nengo.Node(output = port_tag(self.keys, self.ports), size_in = self.dim, size_out = self.dim, label = 'port_tag')
            self.port_val = nengo.Node(output = port_val(self.keys, self.ports), size_in = self.dim, size_out = self.dim, label = 'port_val')
            self.port_swap = nengo.Node(output = port_swap(self.keys, self.ports), size_in = 2*self.dim, size_out = 1, label = 'port_swap')
            self.port_rule = nengo.Node(output = high_rule(), size_in = self.dim, size_out = 1, label = 'port_rule')
            # port_adjust = nengo.Node(output = adjust_port(self.keys, self.ports), size_in = 2*self.dim, size_out = self.dim, label = 'port_adjust')

            nengo.Connection(self.port_dfa.ordered_outputs[0], self.port_new[:self.dim]) 
            nengo.Connection(self.port_dfa.ordered_outputs[1], self.port_new[self.dim:2*self.dim])
            nengo.Connection(self.port_dfa.ordered_outputs[2], self.port_tag)
            nengo.Connection(self.port_dfa.ordered_outputs[3], self.port_val)
            nengo.Connection(self.port_dfa.ordered_outputs[4], self.rule_dfa.statevars.ordered_svs[0].input)
            nengo.Connection(self.port_dfa.ordered_outputs[5], self.rule_dfa.statevars.ordered_svs[1].input)
            nengo.Connection(self.port_dfa.ordered_outputs[6], self.port_swap[:self.dim])
            nengo.Connection(self.port_dfa.ordered_outputs[7], self.port_swap[self.dim:2*self.dim])
            nengo.Connection(self.port_dfa.ordered_outputs[8], self.port_rule)
            #nengo.Connection(self.port_dfa.ordered_outputs[9], node_free)
            #nengo.Connection(self.port_dfa.ordered_outputs[10], node_free)


class Pairs(spa.Network):
    def __init__(self, vocab, theta, keys, pairs, nloc, vloc, label = 'pairs'):
        super().__init__(label = label)
        self.vocab = vocab
        self.theta = theta
        self.dim = vocab.dimensions 
        self.keys = keys
        self.pairs = pairs
        self.nloc = nloc
        self.vloc = vloc

        pair_args = ["PA_NEW","PA_FST","PA_SND","PA_ADJUST","PA_SFLAG","PA_GFLAG", "PA_NULL"]

        pair_statevars = [("command", spa.SemanticPointer),
                          ("pk1_path", spa.SemanticPointer), 
                          ("pk2_path", spa.SemanticPointer), 
                          ("pnew1_out", spa.SemanticPointer),
                          ("pnew2_out", spa.SemanticPointer),
                          ("pfst_out", spa.SemanticPointer),
                          ("psnd_out", spa.SemanticPointer),
                          ("padj_out", spa.SemanticPointer),
                          ("psf_out", spa.SemanticPointer),
                          ("pgf_out", spa.SemanticPointer)
                          ]

        pair_table = {
                (self.vocab["PA_NEW"], None, None):(self.vocab["PA_NULL"], InputVar("key1","pnew1_out"), InputVar("key2","pnew2_out")),
                (self.vocab["PA_FST"], None, None):(self.vocab["PA_NULL"], InputVar("key1","pfst_out")),
                (self.vocab["PA_SND"], None, None):(self.vocab["PA_NULL"], InputVar("key1","psnd_out")),
                (self.vocab["PA_ADJUST"], None, None):(self.vocab["PA_NULL"], InputVar("key1","padj_out")),
                (self.vocab["PA_SFLAG"], None, None):(self.vocab["PA_NULL"], InputVar("key1","psf_out")),
                (self.vocab["PA_GFLAG"], None, None):(self.vocab["PA_NULL"], InputVar("key1","pgf_out"))
                }
        pair_inputs = [
                ("key1", self.dim),
                ("key2", self.dim),
                ]
        pair_outputs = [
                ("pair_new1","pnew1_out"),
                ("pair_new2","pnew2_out"),
                ("pair_fst","pfst_out"),
                ("pair_snd","psnd_out"),
                ("pair_adj","padj_out"),
                ("pair_sfl","psf_out"),
                ("pair_gfl","pgf_out")
                ]
        # given two port keys, put a new keyed pair in dictionary of pairs
        # return the key for pairs 
        def new_pair(keys, pairs):
            def new(t,x):
                port_1 = x[:self.dim]
                port_2 = x[self.dim:2*self.dim]
                empty = hp.check_key_empty(keys, pairs)
                if empty == 0:
                    str_key = str(len(pairs))
                    if self.vocab[f"k_{str_key}"].v:
                        pairs[f"k_{str_key}"] = port_1, port_2
                    else:
                        vocab.populate(f"k_{str_key}")
                        keys.append(f"k_{str_key}")
                        ports[f"k_{str_key}"] = port_1, port_2
                    return vocab[f"k_{str_key}"].v
                elif empty != 0:
                    name = hp.from_vocab(empty, self.vocab)
                    ports[name] = port_1, port_2
                    return empty
            return new
        #given the pair key, return the first ports key
        def pair_fst(keys, pairs):
            def get_fst(t,x):
                key = x
                key_name = hp.from_vocab(key, self.vocab)
                if key_name in keys and key_name in list(pairs.keys()):
                    port1, port2 = ports[key_name]
                    return port1
                else:
                    return np.zeros(self.dim)
            return get_fst
        # given the pair key, return the second ports key
        def pair_snd(keys, pairs):
            def get_snd(t,x):
                key = x
                key_name = hp.from_vocab(key, self.vocab)
                if key_name in keys and key_name in list(pairs.keys()):
                    port1, port2 = ports[key_name]
                    return port2
                else:
                    return np.zeros(self.dim)
            return get_snd
        # ignore for now
        def adjust_pair(keys, pairs): #add ports??
            #need to work out this set of interactions -> likely need some state to hold pair key in memory while ports are individually adjusted
            # or have two nodes which perform adjustments??? -> harder would need to modify table
            def adjust(t,x):
                pass

        # also need to figure out how to set up functions for setting and getting par flag

        with self:
            self.pair_dfa = DFA(pair_statevars, pair_inputs, pair_outputs, pair_table, self.vocab, start=(self.vocab["PA_NULL"], None, None))
            # setting up inputs
            self.key_1 = spa.State(self.vocab, label = 'key 1 in')
            self.key_2 = spa.State(self.vocab, label = 'key 2 in')
            nengo.Connection(self.key_1.output, self.pair_dfa.input_key1)
            nengo.Connection(self.key_2.output, self.pair_dfa.input_key2)

            #setting up outputs
            self.pair_new = nengo.Node(output = new_pair(self.keys, self.pairs), size_in = 2*self.dim, size_out = self.dim, label = 'pair new')
            self.pair_first = nengo.Node(output = pair_fst(self.keys, self.pairs), size_in = self.dim, size_out = self.dim, label = 'pair first')
            self.pair_second = nengo.Node(output = pair_snd(self.keys, self.pairs), size_in = self.dim, size_out = self.dim, label = 'pair second')
            #self.pair_adjust = nengo.Node(output = adjust_pair(self.nodes_dict, self.keys, self.nloc), size_in = self.dim, size_out = self.dim, label = 'node_take')
            #self.pair_sflag = nengo.Node(output = node_free(self.nodes_dict), size_in = self.dim, size_out = 1, label = 'node_free')
            #self.pair_gflag = nengo.Node(output = node_free(self.nodes_dict), size_in = self.dim, size_out = 1, label = 'node_free')

            #print(list(self.pair_dfa.ordered_outputs))


            nengo.Connection(self.pair_dfa.ordered_outputs[0], self.pair_new[:self.dim]) 
            nengo.Connection(self.pair_dfa.ordered_outputs[1], self.pair_new[self.dim:2*self.dim])
            nengo.Connection(self.pair_dfa.ordered_outputs[2], self.pair_first)
            nengo.Connection(self.pair_dfa.ordered_outputs[3], self.pair_second)


class Redexes(spa.Network):
    # this network manages the active redexes
    def __init__(self, vocab, label = 'rbag'):
        self.vocab = vocab
        self.rbag_dict = {}
    pass

class Nodes(spa.Network):
    # this network manages the pairs and ports in GNet 
    def __init__(self, vocab, theta, nodes, keys, nloc, label = 'nodes'):
        super().__init__(label=label)
        self.vocab = vocab
        self.dim = vocab.dimensions
        self.theta = theta
        self.nodes_dict = nodes
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
                (self.vocab["N_CREATE"], None, None): (self.vocab["N_NULL"], InputVar("key", "kcs_out"), InputVar("pair", "pcs_out")),
                (self.vocab["N_STORE"], None, None): (self.vocab["N_NULL"], InputVar("key", "kcs_out"), InputVar("pair", "pcs_out")),
                (self.vocab["N_LOAD"], None, None): (self.vocab["N_NULL"], InputVar("key", "kl_out")),
                (self.vocab["N_EXCHANGE"], None, None): (self.vocab["N_NULL"], InputVar("key", "ke_out"), InputVar("pair", "pe_out")),
                (self.vocab["N_TAKE"], None, None): (self.vocab["N_NULL"], InputVar("key", "kt_out")),
                (self.vocab["N_FREE"], None, None): (self.vocab["N_NULL"], InputVar("key", "kf_out"))
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
        # given a key for where to put in the node_dict, and a pair key -> add pair key to key position (controller manages making the node_dict keys) 
        def node_create(nodes_dict, keys):
            def creator(t,x):
                key = x[:self.dim]
                pair = x[self.dim:2*self.dim]
                #nloc.append(key)
                key_name = hp.from_vocab(key, self.vocab)
                if key_name in keys:
                    nodes_dict[key_name] = pair
                return np.zeros(self.dim)
            return creator
        #given a key return the pair key
        def node_load(nodes_dict, keys, nloc):
            def loader(t,x):
                key = x
                if len(nodes_dict) == 0:
                    return np.zeros(self.dim)
                else:
                    key_name = hp.from_vocab(key, self.vocab)
                    if key_name in keys and key_name in nloc:
                        return nodes_dict[key_name]
                    else:
                        return np.zeros(self.dim)
            return loader
        # connect exchanger to pairs => perform exchange
        # given a key and a pair, exchange the pair key at node_dict[key] and return the original pair key stored at node_dict[key]
        def node_exchange(nodes_dict, keys, nloc):
            def exchanger(t, x):
                key = x[:self.dim]
                pair = x[self.dim:2*self.dim]
                key_name = hp.from_vocab(key, self.vocab)
                if key_name in keys and key_name in nloc:
                    return_pair = nodes_dict[key_name]
                    nodes_dict[key_name] = pair                
                else:
                    return_pair = np.zeros(self.dim)
                return return_pair
            return exchanger
        # given a key, return the pair key stored there, and pop the key
        def node_take(nodes_dict, keys, nloc):
            def taker(t,x):
                key = x
                key_name = hp.from_vocab(key, self.vocab)
                if key_name in keys and key_name in nloc:
                    return_pair = nodes_dict.pop(key_name)
                else:
                    return_pair = np.zeros(self.dim)
                return return_pair
            return taker
        # check if nodes_dict[key] exists, if yes return 1, if no return 0
        def node_free(nodes_dict):
            def freedom(t,x):
                key = x
                key_name = hp.from_vocab(key, self.vocab)
                if nodes_dict[key_name]:
                    return 0
                else:
                    return 1
            return freedom

        with self:
            self.node_dfa = DFA(node_statevars, node_inputs, node_outputs, node_table, self.vocab, start=(self.vocab["N_NULL"], None, None)) 

            # setting up inputs
            self.key_in = spa.State(self.vocab, label = 'key in')
            self.pair_in = spa.State(self.vocab, label = 'pair in')
            nengo.Connection(self.key_in.output, self.node_dfa.input_key)
            nengo.Connection(self.pair_in.output, self.node_dfa.input_pair)

            #setting up outputs
            self.node_create = nengo.Node(output = node_create(self.nodes_dict, self.keys), size_in = 2*self.dim, size_out = self.dim, label = 'node_create')
            self.node_load = nengo.Node(output = node_load(self.nodes_dict, self.keys, self.nloc), size_in = self.dim, size_out = self.dim, label = 'node_load')
            self.node_exchange = nengo.Node(output = node_exchange(self.nodes_dict, self.keys, self.nloc), size_in = 2*self.dim, size_out = self.dim, label = 'node_exchange')
            self.node_take = nengo.Node(output = node_take(self.nodes_dict, self.keys, self.nloc), size_in = self.dim, size_out = self.dim, label = 'node_take')
            self.node_free = nengo.Node(output = node_free(self.nodes_dict), size_in = self.dim, size_out = 1, label = 'node_free')

            #connecting dfa outputs to function ports performing operations on nodes_dict
            nengo.Connection(self.node_dfa.ordered_outputs[0], self.node_create[:self.dim]) 
            nengo.Connection(self.node_dfa.ordered_outputs[1], self.node_create[self.dim:2*self.dim])
            nengo.Connection(self.node_dfa.ordered_outputs[3], self.node_exchange[:self.dim])
            nengo.Connection(self.node_dfa.ordered_outputs[4], self.node_exchange[self.dim:2*self.dim])
            nengo.Connection(self.node_dfa.ordered_outputs[2], self.node_load)
            nengo.Connection(self.node_dfa.ordered_outputs[5], self.node_take)
            nengo.Connection(self.node_dfa.ordered_outputs[6], self.node_free)

# class vars(spa.network):
#     # this network manages variables in gnet 
#     def __init__(self):
#         self.vars_dict = {}
    
class GNET(spa.Network):
    # this handles the representation of the entire graph network 
    def __init__(self, vocab, theta, ports, pairs, keys, nloc, vloc, nodes_dict, label = 'net'):
        super().__init__(label=label)
        self.vocab = vocab
        self.dim = vocab.dimensions
        self.theta = theta
        
        self.nloc = nloc
        self.vloc = vloc
        self.keys = keys
        self.ports = ports
        self.pairs = pairs
        self.node_dict = nodes_dict
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
                (self.vocab["G_NODE"], None, None): (self.vocab["G_NULL"], InputVar("command", "c_outn"), InputVar("key", "k_outn"), InputVar("p_val","p_outn")),
                (self.vocab["G_VAR"], None, None): (self.vocab["G_NULL"], InputVar("command", "c_outv"), InputVar("key", "k_outv"), InputVar("p_val","p_outv"))
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
            self.gnet_dfa = DFA(gnet_statevars, gnet_inputs, gnet_outputs, gnet_table, self.vocab, start=(self.vocab["G_NULL"], None, None)) 

            # setting up inputs
            self.command_in = spa.State(self.vocab, label = 'command in')
            self.key_in = spa.State(self.vocab, label = 'key in')
            self.pair_in = spa.State(self.vocab, label = 'pair/port in')
            nengo.Connection(self.command_in.output, self.gnet_dfa.input_command)
            nengo.Connection(self.key_in.output, self.gnet_dfa.input_key)
            nengo.Connection(self.pair_in.output, self.gnet_dfa.input_p_val)

            self.node_manager = Nodes(self.vocab, self.theta, self.node_dict, self.keys, self.nloc)

            nengo.Connection(self.gnet_dfa.ordered_outputs[0], self.node_manager.node_dfa.statevars.ordered_svs[0].input)
            nengo.Connection(self.gnet_dfa.ordered_outputs[1], self.node_manager.key_in.input)
            nengo.Connection(self.gnet_dfa.ordered_outputs[1], self.node_manager.pair_in.input)

class Def(spa.Network):
    pass

class Book(spa.Network):
    pass 


#with spa.Network() as model:
#    ports = {}
#    pairs = {}
#    keys = []
#    vloc = []
#    nloc = []
#    nodes_dict = {}
#    # need var_dict and rbagloc + rbag_dict
#    gnet = GNET(voc, theta, ports, pairs, keys, nloc, vloc, nodes_dict)
#    ports = Ports(voc, theta, keys, ports, nloc, vloc)
#    pairs = Pairs(voc, theta, keys, pairs, nloc, vloc)
#    # print(gnet.nloc)



