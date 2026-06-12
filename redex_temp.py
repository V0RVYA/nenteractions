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
voc.add("NULL", np.zeros(d))

class Redex_Push(spa.Network):
    def __init__(self, vocab, theta, rbags, keys, label = 'push_redex'):
        super().__init__(label=label)
        self.vocab = vocab
        self.dim = vocab.dimensions
        self.theta = theta
        self.rbag_dict = rbags
        self.keys = keys
        
        # used internally -> should be replaced with spa states holding these values but I don't have time
        self.rpair = []
        self.rports = []
        self.rtags = []

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
                (self.vocab["T_VAR"], voc["T_VAR"], None): (self.vocab["I_LINK"], None, StateVar("key1","rule")),
                (self.vocab["T_VAR"], voc["T_REF"], None): (self.vocab["I_LINK"], None, StateVar("key1","rule")),
                (self.vocab["T_VAR"], voc["T_ERA"], None): (self.vocab["I_LINK"], None, StateVar("key1","rule")),
                (self.vocab["T_VAR"], voc["T_NUM"], None): (self.vocab["I_LINK"], None, StateVar("key1","rule")),
                (self.vocab["T_VAR"], voc["T_CON"], None): (self.vocab["I_LINK"], None, StateVar("key1","rule")),
                (self.vocab["T_VAR"], voc["T_DUP"], None): (self.vocab["I_LINK"], None, StateVar("key1","rule")),
                (self.vocab["T_VAR"], voc["T_OPR"], None): (self.vocab["I_LINK"], None, StateVar("key1","rule")),
                (self.vocab["T_VAR"], voc["T_SWI"], None): (self.vocab["I_LINK"], None, StateVar("key1","rule")),
                (self.vocab["T_REF"], voc["T_VAR"], None): (self.vocab["I_LINK"], None, StateVar("key1","rule")),
                (self.vocab["T_REF"], voc["T_REF"], None): (self.vocab["I_VOID"], None, StateVar("key1","rule")),
                (self.vocab["T_REF"], voc["T_ERA"], None): (self.vocab["I_VOID"], None, StateVar("key1","rule")),
                (self.vocab["T_REF"], voc["T_NUM"], None): (self.vocab["I_VOID"], None, StateVar("key1","rule")),
                (self.vocab["T_REF"], voc["T_CON"], None): (self.vocab["I_CALL"], None, StateVar("key1","rule")),
                (self.vocab["T_REF"], voc["T_DUP"], None): (self.vocab["I_ERAS"], None, StateVar("key1","rule")),
                (self.vocab["T_REF"], voc["T_OPR"], None): (self.vocab["I_CALL"], None, StateVar("key1","rule")),
                (self.vocab["T_REF"], voc["T_SWI"], None): (self.vocab["I_CALL"], None, StateVar("key1","rule")),
                (self.vocab["T_ERA"], voc["T_VAR"], None): (self.vocab["I_LINK"], None, StateVar("key1","rule")),
                (self.vocab["T_ERA"], voc["T_REF"], None): (self.vocab["I_VOID"], None, StateVar("key1","rule")),
                (self.vocab["T_ERA"], voc["T_ERA"], None): (self.vocab["I_VOID"], None, StateVar("key1","rule")),
                (self.vocab["T_ERA"], voc["T_NUM"], None): (self.vocab["I_VOID"], None, StateVar("key1","rule")),
                (self.vocab["T_ERA"], voc["T_CON"], None): (self.vocab["I_ERAS"], None, StateVar("key1","rule")),
                (self.vocab["T_ERA"], voc["T_DUP"], None): (self.vocab["I_ERAS"], None, StateVar("key1","rule")),
                (self.vocab["T_ERA"], voc["T_OPR"], None): (self.vocab["I_ERAS"], None, StateVar("key1","rule")),
                (self.vocab["T_ERA"], voc["T_SWI"], None): (self.vocab["I_ERAS"], None, StateVar("key1","rule")),
                (self.vocab["T_NUM"], voc["T_VAR"], None): (self.vocab["I_LINK"], None, StateVar("key1","rule")),
                (self.vocab["T_NUM"], voc["T_REF"], None): (self.vocab["I_VOID"], None, StateVar("key1","rule")),
                (self.vocab["T_NUM"], voc["T_ERA"], None): (self.vocab["I_VOID"], None, StateVar("key1","rule")),
                (self.vocab["T_NUM"], voc["T_NUM"], None): (self.vocab["I_VOID"], None, StateVar("key1","rule")),
                (self.vocab["T_NUM"], voc["T_CON"], None): (self.vocab["I_ERAS"], None, StateVar("key1","rule")),
                (self.vocab["T_NUM"], voc["T_DUP"], None): (self.vocab["I_ERAS"], None, StateVar("key1","rule")),
                (self.vocab["T_NUM"], voc["T_OPR"], None): (self.vocab["I_OPER"], None, StateVar("key1","rule")),
                (self.vocab["T_NUM"], voc["T_SWI"], None): (self.vocab["I_SWIT"], None, StateVar("key1","rule")),
                (self.vocab["T_CON"], voc["T_VAR"], None): (self.vocab["I_LINK"], None, StateVar("key1","rule")),
                (self.vocab["T_CON"], voc["T_REF"], None): (self.vocab["I_CALL"], None, StateVar("key1","rule")),
                (self.vocab["T_CON"], voc["T_ERA"], None): (self.vocab["I_ERAS"], None, StateVar("key1","rule")),
                (self.vocab["T_CON"], voc["T_NUM"], None): (self.vocab["I_ERAS"], None, StateVar("key1","rule")),
                (self.vocab["T_CON"], voc["T_CON"], None): (self.vocab["I_ANNI"], None, StateVar("key1","rule")),
                (self.vocab["T_CON"], voc["T_DUP"], None): (self.vocab["I_COMM"], None, StateVar("key1","rule")),
                (self.vocab["T_CON"], voc["T_OPR"], None): (self.vocab["I_COMM"], None, StateVar("key1","rule")),
                (self.vocab["T_CON"], voc["T_SWI"], None): (self.vocab["I_COMM"], None, StateVar("key1","rule")),
                (self.vocab["T_DUP"], voc["T_VAR"], None): (self.vocab["I_LINK"], None, StateVar("key1","rule")),
                (self.vocab["T_DUP"], voc["T_REF"], None): (self.vocab["I_ERAS"], None, StateVar("key1","rule")),
                (self.vocab["T_DUP"], voc["T_ERA"], None): (self.vocab["I_ERAS"], None, StateVar("key1","rule")),
                (self.vocab["T_DUP"], voc["T_NUM"], None): (self.vocab["I_ERAS"], None, StateVar("key1","rule")),
                (self.vocab["T_DUP"], voc["T_CON"], None): (self.vocab["I_COMM"], None, StateVar("key1","rule")),
                (self.vocab["T_DUP"], voc["T_DUP"], None): (self.vocab["I_ANNI"], None, StateVar("key1","rule")),
                (self.vocab["T_DUP"], voc["T_OPR"], None): (self.vocab["I_COMM"], None, StateVar("key1","rule")),
                (self.vocab["T_DUP"], voc["T_SWI"], None): (self.vocab["I_COMM"], None, StateVar("key1","rule")),
                (self.vocab["T_OPR"], voc["T_VAR"], None): (self.vocab["I_LINK"], None, StateVar("key1","rule")),
                (self.vocab["T_OPR"], voc["T_REF"], None): (self.vocab["I_CALL"], None, StateVar("key1","rule")),
                (self.vocab["T_OPR"], voc["T_ERA"], None): (self.vocab["I_ERAS"], None, StateVar("key1","rule")),
                (self.vocab["T_OPR"], voc["T_NUM"], None): (self.vocab["I_OPER"], None, StateVar("key1","rule")),
                (self.vocab["T_OPR"], voc["T_CON"], None): (self.vocab["I_COMM"], None, StateVar("key1","rule")),
                (self.vocab["T_OPR"], voc["T_DUP"], None): (self.vocab["I_COMM"], None, StateVar("key1","rule")),
                (self.vocab["T_OPR"], voc["T_OPR"], None): (self.vocab["I_ANNI"], None, StateVar("key1","rule")),
                (self.vocab["T_OPR"], voc["T_SWI"], None): (self.vocab["I_COMM"], None, StateVar("key1","rule")),
                (self.vocab["T_SWI"], voc["T_VAR"], None): (self.vocab["I_LINK"], None, StateVar("key1","rule")),
                (self.vocab["T_SWI"], voc["T_REF"], None): (self.vocab["I_CALL"], None, StateVar("key1","rule")),
                (self.vocab["T_SWI"], voc["T_ERA"], None): (self.vocab["I_ERAS"], None, StateVar("key1","rule")),
                (self.vocab["T_SWI"], voc["T_NUM"], None): (self.vocab["I_SWIT"], None, StateVar("key1","rule")),
                (self.vocab["T_SWI"], voc["T_CON"], None): (self.vocab["I_COMM"], None, StateVar("key1","rule")),
                (self.vocab["T_SWI"], voc["T_DUP"], None): (self.vocab["I_COMM"], None, StateVar("key1","rule")),
                (self.vocab["T_SWI"], voc["T_OPR"], None): (self.vocab["I_COMM"], None, StateVar("key1","rule")),
                (self.vocab["T_SWI"], voc["T_SWI"], None): (self.vocab["I_ANNI"], None, StateVar("key1","rule"))
                }

        rules_inputs = [
                ("dummyin", self.dim),
                ]
        rules_outputs = [
                ("rule_out","rule")
                ]
        redex_push_adds = ["RPU_2", "RPT_1", "RPT_2", "RRULE"]
        hp.add_voc(redex_push_adds, self.vocab)


        # these are gonna interface with a spa.network in ports and pairs called mailbox, to rout the commands appropriately, and return the output to the next step of the process
        # that next step is what the return address does for us
        #TEST
        # want it to when given a pair key, add the pair to rpair (for temp storage during the push action)
        # send the return address(RPU_2), command(PA_FST) and the pair out
        def redex_get_first(rpair, keys):
            def get_fst(t,x):
                pair = x
                key_name = hp.from_voc(pair, self.vocab)
                if key_name in keys:
                    rpair.append(pair)
                    return np.concatenate(self.vocab["RPU_2"].v, self.vocab["PA_FST"].v, pair)
                else:
                    return np.zeros(3*self.dim)
            return get_fst
        #TEST
        # want it when receiving the first port key, add the port key to rport, retrieve the pair from rpair
        # send the return address, command and pair out
        def redex_get_second(rports, rpair, keys):
            def get_snd(t,x):
                port = x
                pair = rpair[0]
                key_name = hp.from_voc(port, self.vocab)
                if key_name in keys:
                    rports.append(port)
                    return np.concatenate(self.vocab["RPT_1"].v, self.vocab["PA_SND"].v, pair)
                else:
                    return np.zeros(3*self.dim)
            return get_snd
        #TEST
        # want is when receiving the second port key, retrieve the first port key from rport, add the second port key to rport
        # send the return address, command and first port key out
        def fst_port_tag(rports, keys):
            def get_tag(t,x):
                port2 = x
                port1 = rports[0]
                key_name = hp.from_voc(port1, self.vocab)
                if key_name in keys:
                    rports[1] = port2
                    return np.concatenate(self.vocab["RPT_2"].v, self.vocab["P_GTAG"].v, port1)
                else:
                    return np.zeros(3*self.dim)
            return get_tag
        #TEST
        # want it when receiving first port tag, store it in rtags, retrieve second port key from rports
        # send out, return address, command and second port key out
        def snd_port_tag(rports, rtags, keys):
            def get_tag(t,x):
                tag1 = x
                port2 = rports[1]
                key_name = hp.from_voc(port2, self.vocab)
                if key_name in keys:
                    rtags[0] = tag1
                    return np.concatenate(self.vocab["RRULE"].v, self.vocab["P_GTAG"].v, port2)
                else:
                    return np.zeros(3*self.dim)
            return get_tag
        #TEST
        # want it to receive second port tag, retrieve first port tag from rtags
        # clear rports and rtags memory, return both tags
        def get_rule(rports, rtags):
            def rule(t,x):
                tag2 = x
                tag1 = rtags[0]
                tags = ["T_VAR","T_REF","T_ERA","T_NUM","T_CON","T_DUP","T_OPR","T_SWI"]
                key1_name = hp.from_voc(tag1, self.vocab)
                key2_name = hp.from_voc(tag2, self.vocab)
                if key1_name in tags and key2_name in tags:
                    rports = []
                    rtags = []
                    return np.concatenate(tag1, tag2)
                else:
                    return np.zeros(2*self.dim)
            return rule
        #TEST
        # want it to receive a rule, retrieve rule string name
        # if string name in high_rules, add to rbag_dict, subdict high (while checking if a new key needs to be generated for the vocab)
        # if string is rule but not in high_rules, add to rbag_dict, subdict low
        # clear rpair, return nothing
        def push(rbag_dict, rpair, keys):
            def push_to(t,x):
                rule = x
                rule_name = hp.from_voc(rule, self.vocab)
                high_rules = ["I_LINK", "I_VOID", "I_ERAS", "I_ANNI"]
                rules = ["I_CALL","I_LINK","I_VOID","I_ERAS","I_COMM","I_ANNI","I_OPER","I_SWIT"]
                if rule_name in rules:
                    if rule_name in high_rules:
                        empty = hp.check_key_empty(keys, rbag_dict["HIGH"])
                        if empty == 0:
                            str_key = str(len(rbag_dicts["HIGH"]))
                            vocab.populate(f"k_{str_key}")
                            keys.append(f"k_{str_key}")
                            rbag_dict["HIGH"][f"k_{str_key}"] = rpair[0]
                            return 0
                        elif empty != 0:
                            name = hp.from_vocab(empty, self.vocab)
                            rbag_dict["HIGH"][name] = rpair[0]
                            return 0
                    else:
                        empty = hp.check_key_empty(keys, rbag_dict["LOW"])
                        if empty == 0:
                            str_key = str(len(rbag_dicts["LOW"]))
                            vocab.populate(f"k_{str_key}")
                            keys.append(f"k_{str_key}")
                            rbag_dict["LOW"][f"k_{str_key}"] = rpair[0]
                            rpair = []
                            return 0
                        elif empty != 0:
                            name = hp.from_vocab(empty, self.vocab)
                            rbag_dict["HIGH"][name] = rpair[0]
                            rpair = []
                            return 0
            return push_to

        with self:
            self.rule_dfa = DFA(rule_statevars, rules_inputs, rules_outputs, rules_table, self.vocab, start=(self.vocab["I_NULL"], self.vocab["I_NULL"], None))
            # setting up inputs
            self.dummy_in = spa.State(self.vocab, label = 'dummyin')
            nengo.Connection(self.dummy_in.output, self.rule_dfa.input_dummyin)

            #setting up outputs
            self.get_fst = nengo.Node(output = redex_get_first(self.rpair, self.keys), size_in = self.dim, size_out = 3*self.dim, label = 'redex_push_get1')
            self.get_snd = nengo.Node(output = redex_get_second(self.rports, self.rpair, self.keys), size_in = self.dim, size_out = 3*self.dim, label = 'redex_push_get2')
            self.fst_tag = nengo.Node(output = fst_port_tag(self.rports, self.keys), size_in = self.dim, size_out = 3*self.dim, label = 'redex_push_tag1')
            self.snd_tag = nengo.Node(output = snd_port_tag(self.rports, self.rtags, self.keys), size_in = self.dim, size_out = 3*self.dim, label = 'redex_push_tag2')
            self.getRule = nengo.Node(output = get_rule(self.rports, self.rtags), size_in = self.dim, size_out = 2*self.dim, label = 'redex_push_rule')
            self.red_psh = nengo.Node(output = push(self.rbag_dict, self.rpair, self.keys), size_in = self.dim, size_out = 1, label = 'redex_push_push')

            #connecting dfa outputs to function ports performing operations on nodes_dict
            nengo.Connection(self.getRule[:self.dim], self.rule_dfa.statevars.ordered_svs[0].input)
            nengo.Connection(self.getRule[self.dim:2*self.dim], self.rule_dfa.statevars.ordered_svs[1].input)
            nengo.Connection(self.rule_dfa.ordered_outputs[0], self.red_psh)
            #print(list(rule_dfa.ordered_outputs))




class Redexes(spa.Network):
    # this network manages the pairs and ports in GNet 
    def __init__(self, vocab, theta, rbags, keys, label = 'redexes'):
        super().__init__(label=label)
        self.vocab = vocab
        self.dim = vocab.dimensions
        self.theta = theta
        self.rbag_dict = rbags # should be of shape {"HIGH":{},"LOW":{}}
        self.keys = keys

        redex_args = ["R_PUSH", "R_POP", "R_NULL", "R_GO"]
        hp.add_voc(redex_args, self.vocab)

        redex_statevars = [("command", spa.SemanticPointer),
                           ("command_node", spa.SemanticPointer),
                           ("path", spa.SemanticPointer),
                           ("push_out", spa.SemanticPointer),
                           ("pop_out", spa.SemanticPointer)
                           ]
        
        redex_table = {
                (self.vocab["R_PUSH"], None, None): (self.vocab["R_NULL"], None, InputVar("pair", "push_out")),
                (self.vocab["R_POP"], None, None): (self.vocab["R_NULL"], self.vocab["R_GO"], StateVar("command_node", "pop_out"))
                }
       
        redex_inputs = [
                ("pair", self.dim),
                ]
        
        redex_outputs = [("to_push", "push_out"),
                         ("to_pop", "pop_out"),
                         ]
        
        #TEST
        #given the go call 
        # split rbag_dict into it's low and high component
        # check they're not empty
        # if high_rbag has nodes, get the last key from it, and return the pair value from rbag_dict["HIGH"] subdict
        # if not, get the last key from low_rbag, and return the pair value from the low subdict
        def pop_redex(rbag_dict):
            def popper(t,x):
                command = x
                high_rbag = rbag_dict["HIGH"]
                low_rbag = rbag_dict["LOW"]
                if len(high_rbag) == 0 and len(low_rbag) == 0:
                    return np.zeros(self.dim)
                else:
                    key_name = hp.from_vocab(command, self.vocab)
                    if key_name == "R_GO":
                        if len(rbag_dict["HIGH"]) != 0:
                            l_key = list(high_rbag)[-1]
                            return rbag_dict["HIGH"].pop(l_key)
                        else:
                            l_key = list(low_rbag)[-1]
                            return rbag_dict["LOW"].pop(l_key)
                    else:
                        return np.zeros(self.dim)
            return popper
        
        with self:
            self.redex_dfa = DFA(redex_statevars, redex_inputs, redex_outputs, redex_table, self.vocab, start=(self.vocab["R_NULL"], None, None)) 

            # setting up inputs
            self.pair_in = spa.State(self.vocab, label = 'pair in')
            nengo.Connection(self.pair_in.output, self.redex_dfa.input_pair)

            #setting up outputs
            self.redex_pop = nengo.Node(output = pop_redex(self.rbag_dict), size_in = self.dim, size_out = self.dim, label = 'redex_popper')
            self.redex_push = Redex_Push(self.vocab, self.theta, self.rbag_dict, self.keys)
 
            #connecting dfa outputs to function ports performing operations on nodes_dict
            nengo.Connection(self.redex_dfa.ordered_outputs[1], self.redex_pop[:self.dim]) 
            nengo.Connection(self.redex_dfa.ordered_outputs[0], self.redex_push.get_fst[:self.dim])


with spa.Network() as model:
    interaction_rules = ["HIGH","LOW"]
    hp.add_voc(interaction_rules, voc)
    keys = []
    rbag_dict = {"HIGH":{}, "LOW":{}}
    redexes = Redexes(voc, theta, rbag_dict, keys)

