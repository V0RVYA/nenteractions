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
        
        with self:
            self.inbox_dfa = DFA(inbox_statevars, inbox_inputs, inbox_outputs, inbox_table, self.vocab, start=(inbox_start)) 
            self.outbox_dfa = DFA(outbox_statevars, outbox_inputs, outbox_outputs, outbox_table, self.vocab, start=(outbox_start))

            conf = nengo.Config(nengo.Ensemble)
            conf[nengo.Ensemble].neuron_type = nengo.neurons.Direct()
            with conf:
                self.in_nodes = []
                for i,v in enumerate(inbox_input_nodes):
                    self.in_nodes.append(spa.State(self.vocab, label = v))
            for inputs, input_nodes in zip(inbox_dict["inputs"], self.in_nodes):
                nengo.Connection()

def direct_conf():
        conf = nengo.Config(nengo.Ensemble)
        conf[nengo.Ensemble].neuron_type = nengo.neurons.Direct()
        return conf

    with direct_conf(): 
        a = spa.State(voc)
        output_states = [spa.State(voc, label=outname) for outname, _ in outputs[:-1]]
        output_states.append(spa.State(len(dfa.output_nodes["strangefruit"]), subdimensions=1, label="strangefruit"))
    nengo.Connection(a.output, dfa.input_a) 

    
    for outnode, state in zip(dfa.ordered_outputs, output_states):
        nengo.Connection(outnode, state.input)


class testbox(mail_box):
    def __init__(self, vocab, theta, indict, outdict, location):
        super().__init__(vocab, theta, indict, outdict, location)
        print(self.inbox_table)
        print(self.outbox_input_nodes)
        print(self.outbox_table)
        print(self.outbox_start)

with spa.Network() as model:
    indict = {"statevars":["test1"], "inputs":["test2"], "outputs":["test3"], "input_nodes":["test4"], "start":(("yolo")), "table":{"yolo2":"test"}}
    outdict = {"statevars":["test1"], "inputs":["test2"], "outputs":["test3"], "input_nodes":["test4"], "start":(("yolo")), "table":{"yolo3":"test2"}}
    test = testbox(voc, theta, indict, outdict, location = "test")


