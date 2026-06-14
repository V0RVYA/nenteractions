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

# inbox and outbox dicts should be of the form: 
# {statevars: list of statevars, inputs: list of inputs, outputs: list of output,statevar tuples, input_nodes: label - in order of inputs,
# start: start argument, table: dfa table}

#location should be a string - naming where the mailbox is

class mail_box(spa.Network):
    def __init__(self, 
                 vocab:spa.Vocabulary, 
                 theta:float, 
                 inbox_dict:dict, 
                 outbox_dict:dict, 
                 location:str, 
                 *args, **kwargs):

        super().__init__(label=f"{location}_mailbox", *args, **kwargs)

        self.vocab              = vocab
        self.theta              = theta

        self.inbox_inputs       = inbox_dict["inputs"]
        self.inbox_statevars    = inbox_dict["statevars"]
        self.inbox_outputs      = inbox_dict["outputs"]
        self.inbox_input_nodes  = inbox_dict["input_nodes"]
        self.inbox_start        = inbox_dict["start"]
        self.inbox_table        = inbox_dict["table"]

        self.outbox_inputs       = outbox_dict["inputs"]
        self.outbox_statevars    = outbox_dict["statevars"]
        self.outbox_outputs      = outbox_dict["outputs"]
        self.outbox_input_nodes  = outbox_dict["input_nodes"]
        self.outbox_start        = outbox_dict["start"]
        self.outbox_table        = outbox_dict["table"]
        
        # print(str_dic(self.inbox_table))
        # print(str_state(self.inbox_start))
        # print(self.inbox_statevars)
        
        # with self:
        #     self.inbox_dfa = DFA(self.inbox_statevars, self.inbox_inputs, self.inbox_outputs, self.inbox_table, self.vocab, start=(self.inbox_start)) 
        #     self.outbox_dfa = DFA(self.outbox_statevars, self.outbox_inputs, self.outbox_outputs, self.outbox_table, self.vocab, start=(self.outbox_start))
        #
        #     conf = nengo.Config(nengo.Ensemble)
        #     conf[nengo.Ensemble].neuron_type = nengo.neurons.Direct()
        #     with conf:
        #         self.in_nodes = []
        #         # self.out_nodes = []
        #         for i,v in enumerate(self.inbox_input_nodes):
        #             self.in_nodes.append(spa.State(self.vocab, label = v))
        #         # for j,c in enumerate(self.outbox_input_nodes):
        #         #     self.out_nodes.append(spa.State(self.vocab, label = c))
        #
        #     # setting up input connections for inbox
        #     for input_dfa, input_nodes in zip(self.inbox_dfa.ordered_inputs, self.in_nodes):
                # nengo.Connection(input_nodes.output, input_dfa)

            # setting up input conncetion for outbox
            # for output_dfa, output_nodes in zip(self.outbox_dfa.ordered_inputs, self.out_nodes):
            #     nengo.Connection(output_nodes.output, output_dfa)


class testbox(mail_box):
    def __init__(self, vocab, theta, indict, outdict, location):
        super().__init__(vocab, theta, indict, outdict, location)
        # print(self.inbox_table)
        # print(self.inbox_start)
        # print(self.inbox_statevars)

with spa.Network() as model:
    voc.populate("Apple;Banana;Cherry;Durian;Elderberry;Fig;Grape;Hawthorn")
    statevars = [("statevar1", spa.SemanticPointer),
                 ("statevar2", spa.SemanticPointer),
                 ("statevar3", spa.SemanticPointer),
                 ("statevar4", int),
                 ("dummyin", spa.SemanticPointer),
                 ("bananapass", spa.SemanticPointer)
                 ]

    table = {
            (voc["Apple"], voc["Banana"], None, 1): (voc["Banana"], voc["Apple"], StateVar("statevar1", "bananapass"), 0), 
            (voc["Banana"], voc["Apple"], None, 0): (voc["Cherry"], voc["Banana"], InputVar("a", "dummyin"), 2),
            (voc["Cherry"], voc["Banana"], None, 2): (voc["Apple"], voc["Banana"], None, 1)
            }

    inputs = [
            ("a", d)
            ]

    outputs = [("fruit", "statevar1"),
               ("otherfruit", "statevar2"),
               ("sometimesfruit", "statevar3"),
               ("strangefruit", "statevar4")
               ]

    input_nodes = ["asshole"]
    start=(voc["Apple"], voc["Banana"], None, 1)

    indict = {"statevars":  statevars, 
              "inputs":     inputs, 
              "outputs":    outputs, 
              "input_nodes":input_nodes, 
              "start":      start, 
              "table":      table}

    def str_state(state):
        repr = []
        for s in state:
            if isinstance(s, spa.SemanticPointer):
                repr.append(f"pointer({s.name})")
            else:
                repr.append(str(s))
        return repr
    
    def str_dic(dic):
        repr = []
        for k, v in dic.items():
            repr.append(f"trigger: {str_state(k)}")
            repr.append(f"next: {str_state(v)}")
        return repr


    # print(str_state(start))
    # print(str_dic(table))

    outdict = {"statevars":["test1"], "inputs":["test2"], "outputs":["test3"], "input_nodes":["test4"], "start":(("yolo")), "table":{"yolo3":"test2"}}
    test = testbox(voc, theta, indict, outdict, location = "test")


