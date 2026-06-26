import nengo
import nengo_spa as spa
import numpy as np
import helper as hp
from neural_dfa import DFA, InputVar, StateVar
from collections import UserDict


class Ports(spa.Network):
    #takes the value of the port and the tag of type as arguments
    def __init__(self, 
                 vocab: spa.Vocabulary, 
                 theta: float, 
                 keys:  list[str], 
                 ports: dict[str, (np.ndarray, np.ndarray)],
                 nloc:  list[spa.SemanticPointer], 
                 vloc:  list[spa.SemanticPointer], 
                 label: str = "ports",
                 sleeptime:float=0.1):
        super().__init__(label = label)
        self.vocab = vocab
        self.dim = vocab.dimensions
        self.theta = theta
        self.sleeptime = sleeptime

        # dictionary that stores the ports 
        self.keys = keys
        self.ports_dict = ports
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

        self.tags = ["T_VAR","T_REF","T_ERA","T_NUM","T_CON","T_DUP","T_OPR","T_SWI"]
        hp.add_voc(self.tags, self.vocab)


        # want:
        # A new entry in the port dictionary containing the tag + value
        # to return the key to the tag/value in the port dictionary
        # tag must come from self.tags or be a 0 vector
        # value can be arbitrary
        # if tag is 0, should do nothing
        # once something has been added, must sleep
        def new_port(keys : list[str], 
                     ports: dict[str, np.ndarray]):
            stopwatch = 0.0
            state = 0
            to_return = np.zeros(self.dim)
            def new(t,x):
                nonlocal stopwatch, state, to_return, keys, ports
                tag = x[:self.dim]
                value = x[self.dim:2*self.dim]
                # This is a sleeping state machine
                # In the 0 state it waits for an input and returns to_return, which is a 0 vector
                # If it recieves an input, enters the 1 state, where it stores a tag, value in memory
                #   then goes to sleep (2 state)
                # In the 2 state it maintains the value it was outputting before, and waits for self.sleeptime ms,
                #   "waking" by retruning to the 0 state and clearing its output at the end
                # The nonlocal variables are necessary to maintain state between function calls
                if state == 0 and tag @ tag >= self.theta:
                    stopwatch = t
                    state = 1
                    empty = hp.check_key_empty(keys, ports) # is a vector or 0
                    tag_name = ""
                    name = ""
                    if empty == 0:
                        str_key = str(len(ports))
                        name = f"K_{str_key}"
                        vocab.populate(name)
                        keys.append(name)
                        ports[name] = tag, value
                        to_return[:] = vocab[name].v
                    elif empty != 0:
                        name = hp.from_vocab(empty, self.vocab)
                        ports[name] = tag, value
                        to_return[:] = empty
                    tag_name = hp.from_vocab(tag, self.vocab)
                    print(f"associating {name} with {tag_name}")
                elif state == 1 and t > stopwatch + self.sleeptime:
                    stopwatch = 0.0
                    state = 2
                elif state == 2 and tag @ tag < self.theta:
                    to_return[:] = 0
                    state = 0
                return to_return
            return new
        #want 
        # given a key(vector) 
        # convert the key to it's string representation => should be of the form "K_XX" with XX being some number -> can be arbitrarily sized
        #want it to retrieve and return the first value of the tuple stored in ports_dict under that key -> the tag of the port
        # return the tag(vector)
        # if input is 0's, or not a key (from error) or not a key in use in ports => return np.zeros(dim)
        def port_tag(keys : list[str], 
                     ports: dict[str, np.ndarray]):
            stopwatch = 0.0
            state = 0
            to_return = np.zeros(self.dim)
            def get_tag(t,x):
                nonlocal keys, ports, stopwatch, state
                key = x
                key_name = hp.from_vocab(key, self.vocab)
                if state == 0 and key_name in ports:
                    stopwatch = t
                    state = 1
                    tag, val = ports[key_name]
                    to_return[:] = tag
                elif state == 1 and t > stopwatch + self.sleeptime:
                    state = 0
                    stopwatch = 0.0
                    to_return[:] = 0
                return to_return
            return get_tag
        # identical behaviour to above but return second value of the tuple stored under the key in ports_dict -> the value of the port (an array)
        # given key to port dict(array) -> retrieve string from vocab 
        # retrieve second value in tuple assigned to the key in port dict as array
        # if input is all 0's or not key or not key in ports_dict -> return all 0 array
        def port_val(keys : list[str], 
                     ports: dict[str, np.ndarray]):
            stopwatch = 0.0
            state = 0
            to_return = np.zeros(self.dim)
            def get_val(t,x):
                nonlocal state, keys, ports, stopwatch, to_return
                key = x
                key_name = hp.from_vocab(key, self.vocab)
                if state == 0 and key_name in ports:
                    stopwatch = t
                    state = 1
                    tag, val = ports[key_name]
                    to_return[:] = val
                elif state == 1 and t > stopwatch + self.sleeptime:
                    state = 0
                    stopwatch = 0.0
                    to_return[:] = 0
                return to_return
            return get_val
            
        #remove -> inside of adjust port -> ignore this
        def port_is_node(keys : list[str], 
                         ports: dict[str, np.ndarray]):
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
        def port_is_var(keys : list[str], 
                        ports: dict[str, np.ndarray]):
            def is_var(t,x):
                key = x
                key_name = hp.from_vocab(key, self.vocab)
                if ports[key_name]:
                    if key_name == "t_var":
                        return 1
                else:
                    return 0
            return is_var
        # given two keys (arrays), want to compare the tags associated with those keys in ports_dict
        # receive both keys, convert them to their string interpretation 
        # retrieve the tag(array) associated with each key -> retrieve the string of the tag from vocab
        # if the tag for key 2, has a higher index in ordered_tags than the tag for key 1 return 1, else return 0
        # if 0's received or not keys received -> return 0
        def port_swap(keys : list[str], 
                      ports: dict[str, np.ndarray]):
            stopwatch = 0.0
            state = 0
            to_return = 0
            def should_swap(t,x):
                nonlocal stopwatch, state, to_return, keys, ports
                key1 = x[:self.dim]
                key2 = x[self.dim:2*self.dim]
                # if both keys are in ordered tags, and if b > a, return 1
                if state == 0 and (key1 @ key1) >= self.theta and (key2 @ key2) >= self.theta:
                    stopwatch = t
                    state = 1
                    key1_name = hp.from_vocab(key1, self.vocab)
                    key2_name = hp.from_vocab(key2, self.vocab)
                    tag1, val1 = self.ports_dict[key1_name]
                    tag2, val2 = self.ports_dict[key2_name]
                    tag1_str = hp.from_vocab(tag1, self.vocab)
                    tag2_str = hp.from_vocab(tag2, self.vocab)
                    a = self.tags.index(tag1_str)
                    b = self.tags.index(tag2_str)
                    if b > a:
                        to_return = 1
                    else:
                        to_return = 0
                if state == 1 and t > stopwatch + self.sleeptime:
                    state = 0
                    stopwatch = 0.0
                return to_return
            return should_swap
        
        def adjust_port(keys :  list[str], 
                        ports:  dict[str, np.ndarray], 
                        nloc:   list[str], 
                        vloc:   list[str]):
            def adjust(t,x):
                # remove is_var and is_node -> logic only used here
                # need to figure out how to perform the logic, of adjusting the pair value to the location of the stored value in nodes/vars
                # replace value with key pointing to value stored somewhere in nodes/vars
                #will wait til functions made to implement
                pass
            return adjust

        with self:
            #self.port_dfa = DFA(port_statevars, port_inputs, port_outputs, port_table, self.vocab, start=(self.vocab["P_NULL"], self.vocab["P_NULL"], None)) 
            #self.rule_dfa = DFA(rule_statevars, rules_inputs, rules_outputs, rules_table, self.vocab, start=(self.vocab["I_NULL"], self.vocab["I_NULL"], None)) 
            # setting up inputs
            conf = nengo.Config(nengo.Ensemble)
            conf[nengo.Ensemble].neuron_type = nengo.neurons.Direct()
            with conf:
                self.trk_in = spa.State(self.vocab, label = 'tag,rule,key in')
                self.vk_in = spa.State(self.vocab, label = 'value,key in')

            # uncomment these
            #nengo.Connection(self.trk_in.output, self.port_dfa.input_trk)
            #nengo.Connection(self.vk_in.output, self.port_dfa.input_vk)

            #setting up outputs
            self.port_new = nengo.Node(output = new_port(self.keys, self.ports_dict), size_in = 2*self.dim, size_out = self.dim, label = 'port_new')
            self.port_tag = nengo.Node(output = port_tag(self.keys, self.ports_dict), size_in = self.dim, size_out = self.dim, label = 'port_tag')
            self.port_val = nengo.Node(output = port_val(self.keys, self.ports_dict), size_in = self.dim, size_out = self.dim, label = 'port_val')
            self.port_swap = nengo.Node(output = port_swap(self.keys, self.ports_dict), size_in = 2*self.dim, size_out = 1, label = 'port_swap')
            # port_adjust = nengo.Node(output = adjust_port(self.keys, self.ports), size_in = 2*self.dim, size_out = self.dim, label = 'port_adjust')

            # uncomment these
            #nengo.Connection(self.port_dfa.ordered_outputs[0], self.port_new[:self.dim]) 
            #nengo.Connection(self.port_dfa.ordered_outputs[1], self.port_new[self.dim:2*self.dim])
            #nengo.Connection(self.port_dfa.ordered_outputs[2], self.port_tag)
            #nengo.Connection(self.port_dfa.ordered_outputs[3], self.port_val)
            #nengo.Connection(self.port_dfa.ordered_outputs[4], self.rule_dfa.statevars.ordered_svs[0].input)
            #nengo.Connection(self.port_dfa.ordered_outputs[5], self.rule_dfa.statevars.ordered_svs[1].input)
            #nengo.Connection(self.port_dfa.ordered_outputs[6], self.port_swap[:self.dim])
            #nengo.Connection(self.port_dfa.ordered_outputs[7], self.port_swap[self.dim:2*self.dim])
            
            # set these up for adjust port when it's ready
            #nengo.Connection(self.port_dfa.ordered_outputs[8], node_free)
            #nengo.Connection(self.port_dfa.ordered_outputs[9], node_free)


class Pairs(spa.Network):
    def __init__(self, 
                 vocab: spa.Vocabulary, 
                 theta: float, 
                 keys:  list[str], 
                 pairs: dict[str, np.ndarray], 
                 ports: dict[str, np.ndarray], 
                 nloc:  list[str], 
                 vloc:  list[str],
                 sleeptimer:float=0.1,
                 label = 'pairs'):
        super().__init__(label = label)
        self.vocab = vocab
        self.theta = theta
        self.dim = vocab.dimensions 
        self.keys = keys
        self.pairs_dict = pairs
        self.ports_dict = ports
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
        # given two port keys -> as arrays, check for key in keys not in use in pairs_dict (this will be the pair key) make new key if none found
        # make new entry in pairs_dict with a pair key(as the dict key), and the two ports keys as it's values
        # then return the key[array] for the pair in pairs_dict
        # if keys received not in ports_dict, or not keys, or all 0's, return np.zeros(self.dim)
        def new_pair(keys, pairs, ports):
            state = 0
            stopwatch = 0.0
            sleeptime = 0.1
            to_return = np.zeros(self.dim)
            def new(t,x):
                nonlocal state, stopwatch, sleeptime, keys, pairs, ports
                port_1 = x[:self.dim]
                port_2 = x[self.dim:2*self.dim]
                port1_name = hp.from_vocab(port_1, self.vocab)
                port2_name = hp.from_vocab(port_2, self.vocab)
                if state == 0 and port1_name in ports and port2_name in ports:
                    state = 1
                elif state == 1:
                    stopwatch = t
                    state = 2
                    empty = hp.check_key_empty(keys, pairs)
                    if empty == 0:
                        str_key = str(len(pairs))
                        if self.vocab[f"k_{str_key}"].v:
                            pairs[f"k_{str_key}"] = port_1, port_2
                        else:
                            vocab.populate(f"k_{str_key}")
                            keys.append(f"k_{str_key}")
                            ports[f"k_{str_key}"] = port_1, port_2
                        to_return[:] = vocab[f"k_{str_key}"].v
                    elif empty != 0:
                        name = hp.from_vocab(empty, self.vocab)
                        ports[name] = port_1, port_2
                        to_return[:] = empty
                elif state == 2 and t > stopwatch + sleeptime:
                    state = 0
                    stopwatch = 0.0
                    to_return[:] = 0
                return to_return
            return new

        #given an input, key(array) -> convert to vocab string representation and check if it's in pair_dict
        # if it is, return the first ports key in the tuple -> as an array
        # if input is not a key in pairs_dict, or is all 0's -> return np.zeros(self.dim)
        def pair_fst(keys, pairs):
            stopwatch = 0.0
            state = 0
            sleeptime = 0.1
            to_return = np.zeros(self.dim)
            def get_fst(t,x):
                nonlocal keys, pairs, stopwatch, state, sleeptime, to_return
                key = x
                key_name = hp.from_vocab(key, self.vocab)
                if state == 0 and key_name in pairs:
                    state = 1
                elif state == 1:
                    stopwatch = t
                    state = 2
                    port1, port2 = pairs[key_name]
                    to_return[:] = port1
                elif state == 2 and t > stopwatch + sleeptime:
                    state = 0
                    stopwatch = 0.0
                    to_return[:] = 0
                return to_return
            return get_fst

        #given an input, key(array) -> convert to vocab string representation and check if it's in pair_dict
        # if it is, return the second ports key in the tuple -> as an array
        # if input is not a key in pairs_dict, or is all 0's -> return np.zeros(self.dim)        def pair_snd(keys, pairs):
            def get_snd(t,x):
                stopwatch = 0.0
                state = 0
                sleeptime = 0.1
                to_return = np.zeros(self.dim)
                def get_fst(t,x):
                    nonlocal keys, pairs, stopwatch, state, sleeptime, to_return
                    key = x
                    key_name = hp.from_vocab(key, self.vocab)
                    if state == 0 and key_name in pairs:
                        state = 1
                    elif state == 1:
                        stopwatch = t
                        state = 2
                        port1, port2 = pairs[key_name]
                        to_return[:] = port2
                    elif state == 2 and t > stopwatch + sleeptime:
                        state = 0
                        stopwatch = 0.0
                        to_return[:] = 0
                return to_return
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
            conf = nengo.Config(nengo.Ensemble)
            conf[nengo.Ensemble].neuron_type = nengo.neurons.Direct()
            with conf:
                self.key_1 = spa.State(self.vocab, label = 'key 1 in')
                self.key_2 = spa.State(self.vocab, label = 'key 2 in')
            nengo.Connection(self.key_1.output, self.pair_dfa.input_key1)
            nengo.Connection(self.key_2.output, self.pair_dfa.input_key2)

            #setting up outputs
            self.pair_new = nengo.Node(output = new_pair(self.keys, self.pairs_dict, self.ports_dict), size_in = 2*self.dim, size_out = self.dim, label = 'pair new')
            self.pair_first = nengo.Node(output = pair_fst(self.keys, self.pairs_dict), size_in = self.dim, size_out = self.dim, label = 'pair first')
            self.pair_second = nengo.Node(output = pair_snd(self.keys, self.pairs_dict), size_in = self.dim, size_out = self.dim, label = 'pair second')
            #self.pair_adjust = nengo.Node(output = adjust_pair(self.nodes_dict, self.keys, self.nloc), size_in = self.dim, size_out = self.dim, label = 'node_take')
            #self.pair_sflag = nengo.Node(output = node_free(self.nodes_dict), size_in = self.dim, size_out = 1, label = 'node_free')
            #self.pair_gflag = nengo.Node(output = node_free(self.nodes_dict), size_in = self.dim, size_out = 1, label = 'node_free')

            #print(list(self.pair_dfa.ordered_outputs))


            nengo.Connection(self.pair_dfa.ordered_outputs[0], self.pair_new[:self.dim]) 
            nengo.Connection(self.pair_dfa.ordered_outputs[1], self.pair_new[self.dim:2*self.dim])
            nengo.Connection(self.pair_dfa.ordered_outputs[2], self.pair_first)
            nengo.Connection(self.pair_dfa.ordered_outputs[3], self.pair_second)



