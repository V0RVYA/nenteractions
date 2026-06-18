import nengo
import nengo_spa as spa
import numpy as np
import helper as hp
from neural_dfa import DFA, InputVar, StateVar
from collections import UserDict

# inbox and outbox dicts should be of the form: 
# {statevars: list of statevars, inputs: list of inputs, outputs: list of output,statevar tuples, input_nodes: label - in order of inputs,
# start: start argument, table: dfa table}

#location should be a string - naming where the mailbox is

class MailBox(spa.Network):
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
            self.inbox_dfa = DFA(self.inbox_statevars, self.inbox_inputs, self.inbox_outputs, self.inbox_table, self.vocab, start=(self.inbox_start)) 
            self.outbox_dfa = DFA(self.outbox_statevars, self.outbox_inputs, self.outbox_outputs, self.outbox_table, self.vocab, start=(self.outbox_start))
        #
            conf = nengo.Config(nengo.Ensemble)
            conf[nengo.Ensemble].neuron_type = nengo.neurons.Direct()
            with conf:
                self.in_nodes = []
                self.out_nodes = []
                for i,v in enumerate(self.inbox_input_nodes):
                    self.in_nodes.append(spa.State(self.vocab, label = v))
                for j,c in enumerate(self.outbox_input_nodes):
                    self.out_nodes.append(spa.State(self.vocab, label = c))
            
            # setting up input connections for inbox
            for input_dfa, input_nodes in zip(self.inbox_dfa.ordered_inputs, self.in_nodes):
                nengo.Connection(input_nodes.output, input_dfa)

            # setting up input conncetion for outbox
            for output_dfa, output_nodes in zip(self.outbox_dfa.ordered_inputs, self.out_nodes):
                nengo.Connection(output_nodes.output, output_dfa)


# class testbox(mail_box):
#     def __init__(self, vocab, theta, indict, outdict, location):
#         super().__init__(vocab, theta, indict, outdict, location)
#

