import nengo
import nengo_spa as spa
import numpy as np
import helper as hp
from neural_dfa import DFA, InputVar, StateVar
from collections import UserDict


class Nodes(spa.Network):
    # this network manages the nodes_dict in GNet 
    def __init__(self, vocab, theta, nodes, keys, nloc, label = 'nodes'):
        super().__init__(label=label)
        self.vocab = vocab
        self.dim = vocab.dimensions
        self.theta = theta
        self.nodes_dict = nodes
        self.keys = keys
        self.nloc = nloc

        node_args = ["N_CREATE", "N_STORE", "N_LOAD", "N_EXCHANGE", "N_TAKE", "N_FREE"]
        hp.add_voc(node_args, self.vocab)

        #defines main table that routes command and key to correct node(performing functions defined below)

        node_statevars = [("command", spa.SemanticPointer),
                          ("k_path", spa.SemanticPointer),
                          ("p_path", spa.SemanticPointer),
                          ("ncs_out", spa.SemanticPointer),
                          ("kl_out", spa.SemanticPointer),
                          ("ke_out", spa.SemanticPointer),
                          ("ne_out", spa.SemanticPointer),
                          ("kt_out", spa.SemanticPointer),
                          ("kf_out", spa.SemanticPointer)
                          ]
        
        node_table = {
                (self.vocab["N_CREATE"],): (self.vocab["NULL"], InputVar("pair", "ncs_out")),
                (self.vocab["N_STORE"],): (self.vocab["NULL"], InputVar("pair", "ncs_out")),
                (self.vocab["N_LOAD"],): (self.vocab["NULL"], InputVar("key", "kl_out")),
                (self.vocab["N_EXCHANGE"],): (self.vocab["NULL"], InputVar("key", "ke_out"), InputVar("pair", "ne_out")),
                (self.vocab["N_TAKE"],): (self.vocab["NULL"], InputVar("key", "kt_out")),
                (self.vocab["N_FREE"],): (self.vocab["NULL"], InputVar("key", "kf_out"))
                }
       
        # input to the Node system expects a 3*dim input [command, location_target(key), pair(key)]
        # command to statevar command
        # rest to following inputs to get passed along
        node_inputs = [
                ("key", self.dim),
                ("pair", self.dim),
                ]
        
        node_outputs = [("node_create", "ncs_out"),
                        ("key_ex","ke_out"),
                        ("node_ex","ne_out"),
                        ("key_load","kl_out"),
                        ("key_take","kt_out"),
                        ("key_free","kf_out"),
                        ]
        # defining node functions

        # key allocation for the node_dict is handled by the controller -> they are added to nloc, and read from nloc to pass to node_create
        # inputs to this node are going to include a key for the pair in pair_dict as an array
        # convert input to their string form
        # if nloc is not empty
        # pop the key for the node from nloc, and add pair to nodes_dict as a value, with the node key as the key 
        #then return 1
        # if the input does not match above criteria, or is all 0, return 0
        def node_create(nodes_dict, keys, nloc):
            state = 0
            stopwatch = 0.0
            sleeptime = 0.1
            to_return = 0
            def creator(t,x):
                nonlocal state, stopwatch, sleeptime, to_return, nodes_dict, keys, nloc
                pair = x 
                if state == 0 and len(nloc) != 0:
                    state =1
                elif state == 1:
                    stopwatch = t
                    state = 2
                    key_name = nloc.pop()
                    nodes_dict[key_name] = pair
                    to_return = 1
                elif state == 2 and t > stopwatch + sleeptime:
                    state = 0
                    stopwatch = 0.0
                    to_return = 0
                return to_return
            return creator

        #given a key (an array) convert it to the string form, and check if its in use by nodes_dict
        # if yes, then return the pair key (array) assigned as a value to the input node key
        # if the array provided does not meet the criteria or if it's all zeros, return an empty erray
        def node_load(nodes_dict, keys):
            state = 0 
            stopwatch = 0.0
            sleeptime = 0.1
            to_return = np.zeros(self.dim)
            def loader(t,x):
                nonlocal state, stopwatch, sleeptime, to_return, nodes_dict, keys
                key = x
                key_name = hp.from_vocab(key, self.vocab)
                if state == 0 and key in nodes_dict:
                    state = 1
                elif state == 1:
                    stopwatch = t
                    state = 2
                    to_return[:] = nodes_dict[key_name]
                elif state == 2 and t > stopwatch + sleeptime:
                    state = 0
                    stopwatch = 0.0
                    to_return[:] = 0
                return to_return
            return loader

        # given a node key and a pair key (as arrays),
        # convert the node key to string, check if it's a valid node_dict key
        # if yes: exchange the pair key at node_dict[key] and return the pair key(array) that was originally stored at node_dict[key]
        # if above criteria aren't met or node key is all zeros -> return 0 array
        def node_exchange(nodes_dict, keys):
            stopwatch = 0.0
            sleeptime = 0.1
            state = 0
            to_return = np.zeros(self.dim)
            def exchanger(t, x):
                nonlocal stopwatch, sleeptime, state, to_return, nodes_dict, keys
                key = x[:self.dim]
                pair = x[self.dim:2*self.dim]
                key_name = hp.from_vocab(key, self.vocab)
                if state == 0 and key_name in nodes_dict:
                    state = 1
                elif state == 1:
                    stopwatch = t
                    state = 2
                    to_return[:] = nodes_dict[key_name]
                    nodes_dict[key_name] = pair                
                elif state == 2 and t > stopwatch + sleeptime:
                    state = 0
                    stopewatch = 0.0
                    to_return[:] = 0
                return to_return
            return exchanger

        # given a node key (as array) as input, convert to its string form
        # if it's a valid key in node_dict then pop the node_dict at that key (removing the entry at that key) and return the pair it stored (an array)
        # if input is not a valid node key, or if it's all 0's, return an empty array
        def node_take(nodes_dict, keys):
            stopwatch = 0.0
            sleeptime = 0.1
            state = 0
            to_return = np.zeros(self.dim)
            def taker(t,x):
                nonlocal stopwatch, sleeptime, state, to_return, nodes_dict, keys
                key = x
                key_name = hp.from_vocab(key, self.vocab)
                if state == 0 and key_name in nodes_dict:
                    state = 1
                elif state == 1:
                    stopwatch = t
                    state = 2
                    to_return[:] = nodes_dict.pop(key_name)
                elif state == 2 and t > stopwatch + sleeptime:
                    state = 0
                    stopwatch = 0.0
                    to_return[:] = 0
                return to_return
            return taker

        # get a key as input (array) -> convert it to string form
        # check if nodes_dict[key] exists, if no return 1, else always return 0
        def node_free(nodes_dict):
            state = 0
            stopwatch = 0.0
            sleeptime = 0.1
            to_return = 0
            def freedom(t,x):
                key = x
                key_name = hp.from_vocab(key, self.vocab)
                if state == 0 and key_name in keys:
                    state = 1
                elif state == 1:
                    stopwatch = t
                    state = 2
                    if key_name not in keys:
                        to_return = 1
                elif state == 2 and t > stopwatch + sleeptime:
                    state = 0
                    stopwatch = 0.0
                    to_return = 0
                return to_return
            return freedom

        with self:
            self.node_dfa = DFA(node_statevars, node_inputs, node_outputs, node_table, self.vocab, start=(self.vocab["NULL"], None, None)) 

            # setting up inputs
            conf = nengo.Config(nengo.Ensemble)
            conf[nengo.Ensemble].neuron_type = nengo.neurons.Direct()
            with conf:
                self.key_in = spa.State(self.vocab, label = 'key in')
                self.pair_in = spa.State(self.vocab, label = 'pair in')
            nengo.Connection(self.key_in.output, self.node_dfa.input_key)
            nengo.Connection(self.pair_in.output, self.node_dfa.input_pair)

            #setting up outputs
            self.node_create = nengo.Node(output = node_create(self.nodes_dict, self.keys, self.nloc), size_in = 2*self.dim, size_out = 1, label = 'node_create')
            self.node_load = nengo.Node(output = node_load(self.nodes_dict, self.keys), size_in = self.dim, size_out = self.dim, label = 'node_load')
            self.node_exchange = nengo.Node(output = node_exchange(self.nodes_dict, self.keys), size_in = 2*self.dim, size_out = self.dim, label = 'node_exchange')
            self.node_take = nengo.Node(output = node_take(self.nodes_dict, self.keys), size_in = self.dim, size_out = self.dim, label = 'node_take')
            self.node_free = nengo.Node(output = node_free(self.nodes_dict), size_in = self.dim, size_out = 1, label = 'node_free')

            #connecting dfa outputs to function ports performing operations on nodes_dict
            nengo.Connection(self.node_dfa.ordered_outputs[0], self.node_create[:self.dim]) 
            nengo.Connection(self.node_dfa.ordered_outputs[2], self.node_exchange[:self.dim])
            nengo.Connection(self.node_dfa.ordered_outputs[3], self.node_exchange[self.dim:2*self.dim])
            nengo.Connection(self.node_dfa.ordered_outputs[1], self.node_load)
            nengo.Connection(self.node_dfa.ordered_outputs[4], self.node_take)
            nengo.Connection(self.node_dfa.ordered_outputs[5], self.node_free)

class Vars(spa.Network):
    # this network manages the vars_dict in GNet 
    def __init__(self, vocab, theta, var, keys, vloc, label = 'vars'):
        super().__init__(label=label)
        self.vocab = vocab
        self.dim = vocab.dimensions
        self.theta = theta
        self.vars_dict = var
        self.keys = keys
        self.vloc = vloc

        var_args = ["V_CREATE", "V_STORE", "V_LOAD", "V_EXCHANGE", "V_TAKE", "V_FREE"]
        hp.add_voc(var_args, self.vocab)

        # this routes the commands for var to the correct var node (performing functions defined below).

        var_statevars = [("command", spa.SemanticPointer),
                          ("k_path", spa.SemanticPointer),
                          ("p_path", spa.SemanticPointer),
                          ("vcs_out", spa.SemanticPointer),
                          ("kl_out", spa.SemanticPointer),
                          ("ke_out", spa.SemanticPointer),
                          ("ve_out", spa.SemanticPointer),
                          ("kt_out", spa.SemanticPointer),
                          ("kf_out", spa.SemanticPointer)
                          ]
        
        var_table = {
                (self.vocab["V_CREATE"],):   (self.vocab["NULL"], InputVar("port", "vcs_out")),
                (self.vocab["V_STORE"],):    (self.vocab["NULL"], InputVar("port", "vcs_out")),
                (self.vocab["V_LOAD"],):     (self.vocab["NULL"], InputVar("key", "kl_out")),
                (self.vocab["V_EXCHANGE"],): (self.vocab["NULL"], InputVar("key", "ke_out"), InputVar("port", "ve_out")),
                (self.vocab["V_TAKE"],):     (self.vocab["NULL"], InputVar("key", "kt_out")),
                (self.vocab["V_FREE"],):     (self.vocab["NULL"], InputVar("key", "kf_out"))
                }
       
        # input to the Node system expects a 3*dim input [command, location_target(key), pair(key)]
        # command to statevar command
        # rest to following inputs to get passed along
        var_inputs = [
                ("key", self.dim),
                ("port", self.dim),
                ]
        
        var_outputs = [("var_create", "vcs_out"),
                       ("key_ex","ke_out"),
                       ("var_ex","ve_out"),
                       ("key_load","kl_out"),
                       ("key_take","kt_out"),
                       ("key_free","kf_out"),
                       ]
        # defining var functions

        # key allocation for the var_dict is handled by the controller -> they are added to vloc, and read from vloc to pass to var_create
        # inputs to this var are going to include a key for the port in port_dict as an array
        # convert input to their string form
        # if vloc is not empty
        # pop the key for the node from vloc, and add por to vars_dict as a value, with the var key as the key 
        #then return 1
        # if the input does not match above criteria, or is all 0, return 0
        def var_create(vars_dict, keys, vloc):
            state = 0
            stopwatch = 0.0
            sleeptime = 0.1
            to_return = 0
            def creator(t,x):
                nonlocal state, stopwatch, sleeptime, to_return, vars_dict, keys, vloc
                port = x 
                if state == 0 and key_name in vloc:
                    state =1
                elif state == 1:
                    stopwatch = t
                    state = 2
                    key_name = vloc.pop()
                    vars_dict[key_name] = pair
                    to_return = 1
                elif state == 2 and t > stopwatch + sleeptime:
                    state = 0
                    stopwatch = 0.0
                    to_return = 0
                return to_return
            return creator

        #given a key (an array) convert it to the string form, and check if its in use by vars_dict
        # if yes, then return the port key (array) assigned as a value to the input node key
        # if the array provided does not meet the criteria or if it's all zeros, return an empty erray
        def var_load(vars_dict, keys):
            state = 0 
            stopwatch = 0.0
            sleeptime = 0.1
            to_return = np.zeros(self.dim)
            def loader(t,x):
                nonlocal state, stopwatch, sleeptime, to_return, vars_dict, keys
                key = x
                key_name = hp.from_vocab(key, self.vocab)
                if state == 0 and key in vars_dict:
                    state = 1
                elif state == 1:
                    stopwatch = t
                    state = 2
                    to_return[:] = vars_dict[key_name]
                elif state == 2 and t > stopwatch + sleeptime:
                    state = 0
                    stopwatch = 0.0
                    to_return[:] = 0
                return to_return
            return loader

        # given a var key and a port/var key (as arrays),
        # convert the var key to string, check if it's a valid var_dict key
        # if yes: exchange the port/var key at var_dict[key] and return the port/var key(array) that was originally stored at var_dict[key]
        # if above criteria aren't met -> return NULL array
        def var_exchange(vars_dict, keys):
            stopwatch = 0.0
            sleeptime = 0.1
            state = 0
            to_return = np.zeros(self.dim)
            def exchanger(t, x):
                nonlocal stopwatch, sleeptime, state, to_return, vars_dict, keys
                key = x[:self.dim]
                pair = x[self.dim:2*self.dim]
                key_name = hp.from_vocab(key, self.vocab)
                if state == 0 and key_name in vars_dict:
                    state = 1
                elif state == 1:
                    stopwatch = t
                    state = 2
                    to_return[:] = vars_dict[key_name]
                    nodes_dict[key_name] = pair                
                elif state == 2 and t > stopwatch + sleeptime:
                    state = 0
                    stopewatch = 0.0
                    to_return[:] = 0
                return to_return
            return exchanger

        # given a var key (as array) as input, convert to its string form
        # if it's a valid key in var_dict then pop the var_dict at that key (removing the entry at that key) and return the pair it stored (an array)
        # if input is not a valid var key, or if it's all 0's, return an empty array
        def var_take(vars_dict, keys):
            stopwatch = 0.0
            sleeptime = 0.1
            state = 0
            to_return = np.zeros(self.dim)
            def taker(t,x):
                nonlocal stopwatch, sleeptime, state, to_return, vars_dict, keys
                key = x
                key_name = hp.from_vocab(key, self.vocab)
                if state == 0 and key_name in vars_dict:
                    state = 1
                elif state == 1:
                    stopwatch = t
                    state = 2
                    to_return[:] = vars_dict.pop(key_name)
                elif state == 2 and t > stopwatch + sleeptime:
                    state = 0
                    stopwatch = 0.0
                    to_return[:] = 0
                return to_return
            return taker

        # get a key as input (array) -> convert it to string form
        # check if vars_dict[key] exists, if no return 1, else always return 0
        def var_free(vars_dict):
            state = 0
            stopwatch = 0.0
            sleeptime = 0.1
            to_return = 0
            def freedom(t,x):
                key = x
                key_name = hp.from_vocab(key, self.vocab)
                if state == 0 and key_name in keys:
                    state = 1
                elif state == 1:
                    stopwatch = t
                    state = 2
                    if key_name not in keys:
                        to_return = 1
                elif state == 2 and t > stopwatch + sleeptime:
                    state = 0
                    stopwatch = 0.0
                    to_return = 0
                return to_return
            return freedom

        with self:
            self.var_dfa = DFA(var_statevars, var_inputs, var_outputs, var_table, self.vocab, start=(self.vocab["NULL"],)) 

            # setting up inputs
            conf = nengo.Config(nengo.Ensemble)
            conf[nengo.Ensemble].neuron_type = nengo.neurons.Direct()
            with conf:
                self.key_in = spa.State(self.vocab, label = 'key in')
                self.port_in = spa.State(self.vocab, label = 'port in')
            nengo.Connection(self.key_in.output, self.var_dfa.input_key)
            nengo.Connection(self.port_in.output, self.var_dfa.input_port)

            #setting up outputs
            self.var_create = nengo.Node(output = var_create(self.vars_dict, self.keys, self.vloc), size_in = 2*self.dim, size_out = 1, label = 'var_create')
            self.var_load = nengo.Node(output = var_load(self.vars_dict, self.keys), size_in = self.dim, size_out = self.dim, label = 'var_load')
            self.var_exchange = nengo.Node(output = var_exchange(self.vars_dict, self.keys), size_in = 2*self.dim, size_out = self.dim, label = 'var_exchange')
            self.var_take = nengo.Node(output = var_take(self.vars_dict, self.keys), size_in = self.dim, size_out = self.dim, label = 'var_take')
            self.var_free = nengo.Node(output = var_free(self.vars_dict), size_in = self.dim, size_out = 1, label = 'var_free')

            #connecting dfa outputs to function ports performing operations on vars_dict
            nengo.Connection(self.var_dfa.ordered_outputs[0], self.var_create[:self.dim]) 
            nengo.Connection(self.var_dfa.ordered_outputs[2], self.var_exchange[:self.dim])
            nengo.Connection(self.var_dfa.ordered_outputs[3], self.var_exchange[self.dim:2*self.dim])
            nengo.Connection(self.var_dfa.ordered_outputs[1], self.var_load)
            nengo.Connection(self.var_dfa.ordered_outputs[4], self.var_take)
            nengo.Connection(self.var_dfa.ordered_outputs[5], self.var_free)

class GEnter(spa.Network):
    # this performs variable substitution during an interaction and it will be the death of me lolololol
    # variable interaction relies on var_dict being populated with chains of variables which point to each other
    # we then go through each link in the chain and delete it, leaving us with one variable pointing to something that is not another variable in vars_dict.
    def __init__(self,
                 vocab: spa.Vocabulary,
                 theta: float,
                 ports: dict,
                 label = "Enter"):
        super().__init__(label = label)
        self.vocab = vocab
        self.dim = vocab.dimensions
        self.theta = theta
        self.ports = ports
        self.var = [] #holds current variable being substituted -> will be used to return final variable being substituted
        self.val = [] #holds value of current variable being substituted -> if it's also a variable will be moved into var and old var will be deleted from var_dict
        
        genter_tags = ["E_GT", "E_CV", "E_XV", "E_CP", "E_NULL", "PORT", "PAIR", "GNET", "P_TAG", "P_VAL"]
        hp.add_voc(genter_tags, self.vocab)
        
        #inbox tracks where information is being received from and what step of the variable substitution process is expecting it
        # passes appropriate key to appropriate places
        inbox_vars = [("radd", spa.SemanticPointer),
                      ("key_pass", spa.SemanticPointer),
                      ("key_out_gt", spa.SemanticPointer),
                      ("key_out_cv", spa.SemanticPointer),
                      ("key_out_xv", spa.SemanticPointer),
                      ("key_out_cp", spa.SemanticPointer)]

        inbox_input = [("key", d)]

        inbox_output = [("get_tag","key_out_gt"),
                        ("check_val","key_out_cv"),
                        ("exchange_val","key_out_xv"),
                        ("check_port","key_out_cp")]

        inbox_table = {(self.vocab["G_ENTER"],):(self.vocab["E_NULL"], InputVar("key", "key_out_gt"),),
                       (self.vocab["E_GT"],):(self.vocab["E_NULL"], InputVar("key", "key_out_gt"),),
                       (self.vocab["E_CV"],):(self.vocab["E_NULL"], InputVar("key", "key_out_cv"),),
                       (self.vocab["E_XV"],):(self.vocab["E_NULL"], InputVar("key", "key_out_xv"),),
                       (self.vocab["E_CP"],):(self.vocab["E_NULL"], InputVar("key", "key_out_cp"),),
                       }
        # paths => let us passthrough states and inputs to their correct outputs
        # port outs go to ports handler, pair outs go to pairs handler
        # gnet + gcommand + inside the house call go to GNET  
        outbox_vars = [("add", spa.SemanticPointer),
                       ("inside_pass", spa.SemanticPointer),
                       ("radd_pass", spa.SemanticPointer),
                       ("gcommand_pass", spa.SemanticPointer),
                       ("command_pass", spa.SemanticPointer),
                       ("key1_pass", spa.SemanticPointer),
                       ("key2_pass", spa.SemanticPointer),
                       ("inside_out", spa.SemanticPointer),
                       ("radd_port_out", spa.SemanticPointer),
                       ("radd_pair_out", spa.SemanticPointer),
                       ("radd_gnet_out", spa.SemanticPointer),
                       ("gcommand_out",spa.SemanticPointer),
                       ("command_pair_out", spa.SemanticPointer),
                       ("command_port_out", spa.SemanticPointer),
                       ("command_gnet_out", spa.SemanticPointer),
                       ("key1_port_out", spa.SemanticPointer),
                       ("key1_pair_out",spa.SemanticPointer),
                       ("key1_gnet_out", spa.SemanticPointer),
                       ("key2_gnet_out", spa.SemanticPointer)
                       ]

        outbox_input = [("radd", d),
                        ("inside", d),
                        ("gcommand", d),
                        ("command",d),
                        ("key1", d),
                        ("key2", d)]

        outbox_output = [("inside_house","inside_out"),
                         ("radd_ports","radd_port_out"),
                         ("radd_pairs","radd_pair_out"),
                         ("radd_gnet","radd_gnet_out"),
                         ("gnet_command","gcommand_out"),
                         ("command_ports","command_port_out"),
                         ("command_pairs","command_pair_out"),
                         ("command_gnet","command_gnet_out"),
                         ("key1_pair","key1_pair_out"),
                         ("key1_port","key1_port_out"),
                         ("key1_gnet","key1_gnet_out"),
                         ("key2_gnet","key2_gnet_out")]
        # might run into issue below with gnet one, cause var exchange and var take have different amounts of input needs, but gnet inbox should handle it, fingers crossed, prayers out.
        outbox_table = {(self.vocab["PAIR"],):(self.vocab["E_NULL"], InputVar("radd", "radd_pair_out"), InputVar("command","command_pair_out"),InputVar("key1","key1_pair_out")),
                        (self.vocab["GNET"],):(self.vocab["E_NULL"], InputVar("inside", "inside_out"), InputVar("radd", "radd_gnet_out"), InputVar("gcommand","gcommand_out"), InputVar("command","command_gnet_out"), InputVar("key1","key1_gnet_out"), InputVar("key2","key2_gnet_out")),
                        (self.vocab["PORT"],):(self.vocab["E_NULL"], InputVar("radd", "radd_port_out"), InputVar("command","command_port_out"),InputVar("key1","key1_port_out"))
                       }
        # receive a port key from gnet (as array)
        # check it's a valid port key, if yes output a command for port to get tag 4*dim arry = PORT + E_CV(return address) + P_TAG (port command) + port_key
        # else return 4*dim of zeros
        def e_get_tag(ports_dict):
            state = 0
            sleeptime = 0.1
            stopwatch = 0.0
            to_return = np.zeros(4*self.dim)
            def get_tag(t,x):
                nonlocal state, sleeptime, stopwatch, to_return, ports_dict
                port_key = x[:self.dim]
                key_str = hp.from_vocab(port_key, self.vocab)
                if state == 0 and key_str in ports_dict.keys:
                    state = 1
                elif state == 1:
                    stopwatch = t
                    state = 2
                    to_return[:] = np.concatenate[(self.vocab["PORT"].v, self.vocab["E_CV"].v, self.vocab["P_TAG"].v, port_key)]
                elif state == 2 and t > stopwatch + sleeptime:
                    state = 0
                    stopwatch = 0.0
                    to_return[:] = np.zeros(4*self.dim)
                return to_return
            return enter

        #check_tag is a DFA that ends the substitution process if the tag of the port held in var_dict is not var -> no variable to substitute. 
        check_tag_statevars = [("tag", spa.SemanticPointer),
                               ("command", spa.SemanticPointer),
                               ("command_path", spa.SemanticPointer),
                               ("command_yes", spa.SemanticPointer),
                               ("command_no", spa.SemanticPointer)]

        check_tag_input = []

        check_tag_output = [("go","command_yes"),
                            ("no_go","command_no")]

        check_tag_args = ["T_VAR","T_REF","T_ERA","T_NUM","T_CON","T_DUP","T_OPR","T_SWI", "CHK_NULL", "GO", "NO_GO"]
        hp.add_voc(check_tag_args, self.vocab)

        check_tag_table = {(self.vocab["T_VAR"],):(self.vocab["CHK_NULL"], self.vocab["GO"], StateVar("command", "command_yes")),
                           (self.vocab["T_REF"],):(self.vocab["CHK_NULL"], self.vocab["NO_GO"], StateVar("command", "command_no")),
                           (self.vocab["T_ERA"],):(self.vocab["CHK_NULL"], self.vocab["NO_GO"], StateVar("command", "command_no")),
                           (self.vocab["T_NUM"],):(self.vocab["CHK_NULL"], self.vocab["NO_GO"], StateVar("command", "command_no")),
                           (self.vocab["T_CON"],):(self.vocab["CHK_NULL"], self.vocab["NO_GO"], StateVar("command", "command_no")),
                           (self.vocab["T_DUP"],):(self.vocab["CHK_NULL"], self.vocab["NO_GO"], StateVar("command", "command_no")),
                           (self.vocab["T_OPR"],):(self.vocab["CHK_NULL"], self.vocab["NO_GO"], StateVar("command", "command_no")),
                           (self.vocab["T_SWI"],):(self.vocab["CHK_NULL"], self.vocab["NO_GO"], StateVar("command", "command_no")),
                           (self.vocab["CHK_NULL"],):(self.vocab["CHK_NULL"], self.vocab["NO_GO"], StateVar("command", "command_no"))
                           }
        # if tag is not var, or var val is none -> this exits the substitution process and returns the port key held in var
        # logic is handled by check_var and check_val
        # if it receives a no_go command (as array) => will end the substitution process, pop the only value in the var dict (an array port key) and return it to gnet outbox 
        # else it passes zeros of size dim
        def exit(var, val):
            state = 0
            stopwatch = 0.0
            sleeptime = 0.1
            to_return = np.zeros(self.dim)
            def breaker(t,x):
                nonlocal var, val, state, stopwatch, sleeptime, to_return
                command = x 
                command_str = hp.from_vocab(command, self.vocab)
                if state == 0 and command_str == "NO_GO":
                    state = 1
                if state == 1:
                    stopwatch = t
                    state = 2
                    val = []
                    to_return[:] = var.pop()
                if state == 2 and t > stopwatch + sleeptime:
                    state = 0
                    stopwatch = 0.0
                    to_return[:] = 0
                return to_return
            return breaker
        # if this receives the go command from check_tag dfa (array size dim)
        # takes the port key from var, and packages it into a command of size 4*dim = PORT + E_XV + P_VAL + port(key)
        # sends it to ports to retrieve value through the outbox 
        # else if zeros or something else received -> return 4*dim array of zeros.
        def check_tag2(var):
            state = 0
            stopwatch = 0.0
            sleeptime = 0.1
            to_return = np.zeros(4*self.dim)
            def make_command(t,x):
                nonlocal var, state, stopwatch, sleeptime, to_return
                command = x
                command_str = hp.from_vocab(command, self.vocab)
                if state == 0 and command_str == "GO":
                    state = 1
                if state == 1:
                    stopwatch = t
                    state = 2
                    port = var[0]
                    to_return[:] = np.concatenate[(self.vocab["PORT"].v, self.vocab["E_XV"].v, self.vocab["P_VAL"].v, port)]
                if state == 2 and t > stopwatch + sleeptime:
                    state = 0
                    sleeptime = 0.0
                    to_return[:] = 0
                return to_return
            return make_command 

        def var_exchange(var, val):
            state = 0
            stopwatch = 0.0
            sleeptime = 0.1
            to_return = np.zeros(6*self.dim)
            def make_command(t,x):
                nonlocal var, state, stopwatch, sleeptime, to_return
                value = x
                if state == 0 and if val.dot(self.vocab["NULL"].v, value) =< self.theta:
                    state = 1
                if state == 1:





class GNET(spa.Network):
    # this handles the representation of the entire graph network 
    # holds node_handler, vars-handler and genter handler (does the variable substitution)
    # has the most complex kind of mailbox.
    #need to add genter_handler into it once genter is done
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
            conf = nengo.Config(nengo.Ensemble)
            conf[nengo.Ensemble].neuron_type = nengo.neurons.Direct()
            with conf:
                self.command_in = spa.State(self.vocab, label = 'command in')
                self.key_in = spa.State(self.vocab, label = 'key in')
                self.pair_in = spa.State(self.vocab, label = 'pair/port in')
            nengo.Connection(self.command_in.output, self.gnet_dfa.input_command)
            nengo.Connection(self.key_in.output, self.gnet_dfa.input_key)
            nengo.Connection(self.pair_in.output, self.gnet_dfa.input_p_val)

            self.node_manager   = Nodes(self.vocab, self.theta, self.node_dict, self.keys, self.nloc)
            self.var_manager    = Vars(self.vocab, self.theta, self.node_dict, self.keys, self.vloc)

            nengo.Connection(self.gnet_dfa.ordered_outputs[0], self.node_manager.node_dfa.statevars.ordered_svs[0].input)
            nengo.Connection(self.gnet_dfa.ordered_outputs[1], self.node_manager.key_in.input)
            nengo.Connection(self.gnet_dfa.ordered_outputs[2], self.node_manager.pair_in.input)
            nengo.Connection(self.gnet_dfa.ordered_outputs[3], self.var_manager.var_dfa.statevars.ordered_svs[0].input)
            nengo.Connection(self.gnet_dfa.ordered_outputs[4], self.var_manager.key_in.input)
            nengo.Connection(self.gnet_dfa.ordered_outputs[5], self.var_manager.port_in.input)



with spa.Network() as model:
    #dimentions and default vocabulary initialized here
    d = 128
    theta = 0.3
    voc = spa.Vocabulary(d)
    voc.add("NULL", np.zeros(d))

    ports = {}
    pairs = {}
    keys = []
    vloc = []
    nloc = []
    nodes_dict = {}
    # need var_dict and rbagloc + rbag_dict
    gnet = GNET(voc, theta, ports, pairs, keys, nloc, vloc, nodes_dict)
#    ports = Ports(voc, theta, keys, ports, nloc, vloc)
#    pairs = Pairs(voc, theta, keys, pairs, nloc, vloc)
#    # print(gnet.nloc)



