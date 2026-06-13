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
    def __init__(self, vocab, theta, rbags, pairs, ports, keys, label = 'push_redex'):
        super().__init__(label=label)
        self.vocab = vocab
        self.dim = vocab.dimensions
        self.theta = theta
        self.rbag_dict = rbags
        self.keys = keys
        self.pairs_dict = pairs
        self.ports_dict = ports
        
        # used internally -> should be replaced with spa states holding these values but I don't have time
        self.rpair = []
        self.rports = []
        self.rtags = []

        self.tags = ["T_VAR","T_REF","T_ERA","T_NUM","T_CON","T_DUP","T_OPR","T_SWI"]

        self.interaction_rules = ["I_CALL","I_LINK","I_VOID","I_ERAS","I_COMM","I_ANNI","I_OPER","I_SWIT", "I_NULL"]
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

        rules_inputs = []
        rules_outputs = [
                ("rule_out","rule")
                ]
        redex_push_adds = ["RPU_2", "RPT_1", "RPT_2", "RRULE"]
        hp.add_voc(redex_push_adds, self.vocab)


        # these are gonna interface with a spa.network in ports and pairs called mailbox, to rout the commands appropriately, and return the output to the next step of the process
        # that next step is what the return address does for us
        #TEST
        # want it to when given a pair key(array), get the string rep of it, if its in pairs add the pair to rpair as array (for temp storage during the push action)
        # send a 3*dim array containing the return address(RPU_2), command(PA_FST) and the pair out
        # if input requirements not met, just return array of zeros
        def redex_get_first(rpair, pair_dict):
            state = 0
            stopwatch = 0.0
            sleeptimer = 0.1
            to_return = np.zeros(3*self.dim)
            def get_fst(t,x):
                nonlocal rpair, pair_dict, state, stopwatch, sleeptimer, to_return
                pair = x
                key_name = hp.from_vocab(pair, self.vocab)
                if state == 0 and key_name in pair_dict:
                    state = 1
                if state == 1:
                    stopwatch = t
                    state = 2
                    rpair.append(pair)
                    to_return[:] = np.concatenate(self.vocab["RPU_2"].v, self.vocab["PA_FST"].v, pair)
                elif state == 2 and t > stopwatch + sleeptimer:
                    state = 0
                    stopwatch = 0.0
                    to_return[:] = 0
                return to_return
            return get_fst
        #TEST
        # receives the first port key as input (array), check that it's a valid key in ports_dict and that rpair is not empty
        # if yes add the port key to rport(temp storage for duration of push), retrieve the pair from rpair
        # send the return address, command and pair out as a 3*self.dim array
        #else send a zero array of size 3*self.dim
        def redex_get_second(rports, rpair, port_dict):
            state = 0
            stopwatch = 0.0
            sleeptimer = 0.1
            to_return = np.zeros(3*self.dim)
            def get_snd(t,x):
                nonlocal rports, rpair, port_dict, states, stopwatch, sleeptimer, to_return
                port = x
                port_name = hp.from_vocab(port, self.vocab)
                if state == 0 and port_name in port_dict and len(rpair) != 0:
                    state = 1
                elif state == 1: 
                    stopwatch = t
                    state = 2
                    pair = rpair[0]
                    rports.append(port)
                    to_return[;] = np.concatenate(self.vocab["RPT_1"].v, self.vocab["PA_SND"].v, pair)
                elif state == 2 and t > stopwatch + sleeptimer:
                    state = 0
                    stopwatch = 0.0
                    to_return[:] = 0
                return to_return
            return get_snd
        #TEST
        # receiving the second port key (array) as input, convert to string form and test that it is a key in ports_dicts and that len(rports) is not 0
        # if yes: retrieve the first port key from rport, add the second port key to rport
        # then return a 3*dim array containing the return address, command and first port key
        # if above criteria are not met, return 3*dim array of 0s
        def fst_port_tag(rports, ports_dicts):
            state = 0
            stopwatch = 0.0
            sleeptimer = 0.1
            to_return = np.zeros(3*self.dim)
            def get_tag(t,x):
                nonlocal rports, ports_dicts, states, stopwatch, sleeptimer, to_return
                port2 = x
                port2_name = hp.from_vocab(port2, self.vocab)
                if state == 0 and port2_name in ports_dicts and len(rports) != 0:
                    state = 1
                elif state == 1:
                    stopwatch = t
                    state = 2
                    port1 = rports[0]
                    rports[1] = port2
                    to_return[:] np.concatenate(self.vocab["RPT_2"].v, self.vocab["P_GTAG"].v, port1)
                elif state = 2 and t > stopwatch + sleeptimer:
                    state = 0
                    stopwatch = 0.0
                    to_return[:] = 0
            return get_tag
        #TEST
        # want it when receiving first port tag (array),convert to string and check that it is in the tag list and check that len(rports) == 2 and len(rtags) = 0
        # if yes: store it in rtags (as array), retrieve second port key(array) from rports
        # send out, a 3*dim array containing return address, command and second port key
        # if above criteria not met return 3*dim array of 0's
        def snd_port_tag(rports, rtags):
            state = 0
            stopwatch = 0.0
            sleeptimer = 0.1
            to_return = np.zeros(3*self.dim)
            def get_tag(t,x):
                nonlocal rports, rtags, states, stopwatch, sleeptimer, to_return 
                tag1 = x
                tag1_name = hp.from_vocab(tag1, self.vocab)
                if state == 0 and tag1_name in self.tags and len(rports) == 2 and len(rtags) == 0:
                    state = 1
                if state = 1:
                    stopwatch = t
                    state = 2
                    port2 = rports[1]
                    rtags.append(tag1)
                    to_return np.concatenate(self.vocab["RRULE"].v, self.vocab["P_GTAG"].v, port2)
                elif state = 2 and t > stopwatch + sleeptimer:
                    state = 0
                    stopwatch = 0.0
                    to_return[:] = 0
                return to_return
            return get_tag
        #TEST
        # want it to receive second port tag (array), convert to string, check it's in list of tags and check that len of rtags == 1
        # if yes: retrieve first port tag(array) from rtags
        # clear rports and rtags memory lists, return both tags as an array size 2*dim
        # if above criteria not met, return 2*dim array of 0's
        def get_rule(rports, rtags):
            state = 0
            stopwatch = 0.0
            sleeptimer = 0.1
            to_return = np.zeros(2*self.dim)
            def rule(t,x):
                nonlocal rports, rtags, state, stopwatch, sleeptimer, to_return
                tag2 = x
                tag2_name = hp.from_vocab(tag2, self.vocab)
                if state == 0 and tag2_name in self.tags len(rtags) == 1:
                    state = 1
                elif state = 1:
                    stopwatch = t
                    state = 2
                    tag1 = rtags[0]
                    to_return[:] = np.concatenate(tag1, tag2)
                elif state = 2 and t > stopwatch + sleeptimer:
                    state = 0
                    stopwatch = 0.0
                    to_return[:] = 0
                return to_return
            return rule
        #TEST
        # want it to receive a rule(array), retrieve rule string name check if it's in rules
        # if criteria met return 1, and then:
        # if string name in high_rules, add to rbag_dict, subdict high (while checking if a new key needs to be generated for the vocab)
        # if string is rule but not in high_rules, add to rbag_dict, subdict low (while checking if we need a new key in vocab)
        # clear rpair, return nothing
        def push(rbag_dict, rpair, keys):
            state = 0
            stopwatch = 0.0
            sleeptime = 0.1
            to_return = 0
            def push_to(t,x):
                nonlocal rbag_dict, rpair, keys, 
                rule = x
                rule_name = hp.from_vocab(rule, self.vocab)
                high_rules = ["I_LINK", "I_VOID", "I_ERAS", "I_ANNI"]
                if state == 0 and rule_name in self.interaction_rules:
                    state = 1
                if state = 1:
                    stopwatch = t
                    state = 2
                    to_return = 1
                    if rule_name in high_rules:
                        empty = hp.check_key_empty(keys, rbag_dict["HIGH"])
                        if empty == 0:
                            str_key = str(len(rbag_dict["HIGH"]))
                            vocab.populate(f"k_{str_key}")
                            keys.append(f"k_{str_key}")
                            rbag_dict["HIGH"][f"k_{str_key}"] = rpair[0]
                        elif empty != 0:
                            name = hp.from_vocab(empty, self.vocab)
                            rbag_dict["HIGH"][name] = rpair[0]
                    else:
                        empty = hp.check_key_empty(keys, rbag_dict["LOW"])
                        if empty == 0:
                            str_key = str(len(rbag_dicts["LOW"]))
                            vocab.populate(f"k_{str_key}")
                            keys.append(f"k_{str_key}")
                            rbag_dict["LOW"][f"k_{str_key}"] = rpair[0]
                            rpair = []
                        elif empty != 0:
                            name = hp.from_vocab(empty, self.vocab)
                            rbag_dict["HIGH"][name] = rpair[0]
                            rpair = []
                elif state = 2 and t > stopwatch + sleeptimer:
                    state = 0
                    stopwatch = 0.0
                    to_return[:] = 0
                return to_return
            return push_to

        with self:
            self.rule_dfa = DFA(rule_statevars, rules_inputs, rules_outputs, rules_table, self.vocab, start=(self.vocab["I_NULL"], self.vocab["I_NULL"], None))
            # setting up inputs

            #setting up outputs
            self.get_fst = nengo.Node(output = redex_get_first(self.rpair, self.pairs_dict), size_in = self.dim, size_out = 3*self.dim, label = 'redex_push_get1')
            self.get_snd = nengo.Node(output = redex_get_second(self.rports, self.rpair, self.ports_dict), size_in = self.dim, size_out = 3*self.dim, label = 'redex_push_get2')
            self.fst_tag = nengo.Node(output = fst_port_tag(self.rports, self.ports_dict), size_in = self.dim, size_out = 3*self.dim, label = 'redex_push_tag1')
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
    def __init__(self, vocab, theta, rbags, keys, ports, pairs, label = 'redexes'):
        super().__init__(label=label)
        self.vocab = vocab
        self.dim = vocab.dimensions
        self.theta = theta
        self.rbag_dict = rbags # should be of shape {"HIGH":{},"LOW":{}}
        self.keys = keys
        self.ports_dict = ports
        self.pairs_dict = pairs

        redex_args = ["R_PUSH", "R_POP", "R_NULL", "R_GO", "R_DONE"]
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
        #given the go call (array) -> convert to string, check if it's "R_GO" 
        # -> split rbag_dict into it's low and high component
        # -> check they're not empty -> if both are empty return array for "R_DONE" (all interactions complete)
        # if high_rbag has nodes, get the last key from it, and return the pair key from rbag_dict["HIGH"] subdict as array while popping it from the rbag dictionary
        # if not, get the last key from low_rbag, and return the pair key from the low subdict while popping it from the rbag dictionary
        # if none of above conditions are met -> return all 0's in array
        def pop_redex(rbag_dict):
            state = 0
            stopwatch = 0.0
            sleeptime = 0.1
            to_return = np.zeros(self.dim)
            def popper(t,x):
                nonlocal rbag_dict, state, sleeptime, stopwatch, to_return
                command = x
                command_name = hp.from_vocab(command, self.vocab)
                if state == 0 and command_name == "R_GO":
                    state = 1
                if state == 1:
                    stopwatch = t
                    state = 2
                    high_rbag = rbag_dict["HIGH"]
                    low_rbag = rbag_dict["LOW"]
                    if len(high_rbag) == 0 and len(low_rbag) == 0:
                        to_return [:] = self.vocab["R_DONE"].v    
                    elif len(rbag_dict["HIGH"]) != 0:
                        l_key = list(high_rbag)[-1]
                        to_return[:] = rbag_dict["HIGH"].pop(l_key)
                    else:
                        l_key = list(low_rbag)[-1]
                        to_return[:] = rbag_dict["LOW"].pop(l_key)
                elif state = 2 and t > stopwatch + sleeptimer:
                    state = 0
                    stopwatch = 0.0
                    to_return[:] = 0
                return to_return
            return popper
        
        with self:
            self.redex_dfa = DFA(redex_statevars, redex_inputs, redex_outputs, redex_table, self.vocab, start=(self.vocab["R_NULL"], None, None)) 

            # setting up inputs
            self.pair_in = spa.State(self.vocab, label = 'pair in')
            nengo.Connection(self.pair_in.output, self.redex_dfa.input_pair)

            #setting up outputs
            self.redex_pop = nengo.Node(output = pop_redex(self.rbag_dict), size_in = self.dim, size_out = self.dim, label = 'redex_popper')
            self.redex_push = Redex_Push(self.vocab, self.theta, self.rbag_dict, self.ports_dict, self.pairs_dict, self.keys)
 
            #connecting dfa outputs to function ports performing operations on nodes_dict
            nengo.Connection(self.redex_dfa.ordered_outputs[1], self.redex_pop) 
            nengo.Connection(self.redex_dfa.ordered_outputs[0], self.redex_push.get_fst)


with spa.Network() as model:
    interaction_rules = ["HIGH","LOW"]
    hp.add_voc(interaction_rules, voc)
    keys = []
    rbag_dict = {"HIGH":{}, "LOW":{}}
    redexes = Redexes(voc, theta, rbag_dict, keys)

