import nengo
import nengo_spa as spa
import numpy as np

from neural_dfa import DFA, InputVar, StateVar 

#dimentions and default vocabulary initialized here
d = 256 
voc = spa.Vocabulary(d)

def Port (spa.Network):
    #takes the value of the port and the tag of type as arguments
    def __init__(self, tag, val):
        pass
    pass

def Pair (self, one, two):
    #takes two ports as an argument and stores them as a tuple
    pass
# voc.populate

def rbag(UserDict):
    # this dictionary acts as a buffer to hold the current set of active redexes
    def __init__(self, pair):
        pass
    pass

def node(UserDict):
    # this dictionary acts as a buffer to store all current nodes 
    def __init__(self, pair):
        pass
    pass 

def var(UserDict):
    # this dictionary acts as a buffer that stores all the ports of the network 
    def __init__(self, port):
        pass
    pass

def GNET(UserDict):
    # this dictionary stores the nodes, redexes, and variables of a single network 
    def __init__(self, node, var, rbag):
        pass
    pass



# Aiming to recreate HVM2 implementation in spiking neural networks
# Below is a summary of relevant components of HVM2 for the task from the HVM paper draft 
# DATA FORMATTING 

# Interaction Combinator system -> represented by interaction calculus system 
# seven types of agents (nodes) and variables -> both create tree 
# nullary nodes[1 main/0 auxillary - ports] vs binary nodes [1 main/2 aux - ports]

#<Node> ::=
    # | "*"                       -- (ERA)ser - nullary
    # | "@" <alphanumeric>        -- (REF)erence - nullary -> immutable net expanded in single interaction
    # | <Numeric>                 -- (NUM)eric - nullary -> performance (optional)
    # | "(" <Tree> <Tree> ")"     -- (CON)structor
    # | "{" <Tree> <Tree> "}"     -- (DUP)licator
    # | "$(" <Tree> <Tree> ")"    -- (OPE)rator -> performance (optional)
    # | "?(" <Tree> <Tree> ")"    -- (SWI)tch -> performance (optional)
#<Tree> ::=
    # | <alphanumeric>            -- (VAR)iable
    # | <Node> 

# Arbitrary nodes (including VAR)       A,B,D,D := <Tree>
# Binary Nodes                          (),{}   := CON|DUP|OPE|SWI
# Nullary Nodes                         O,o     := ERA|REF|NUM 
# Numeric Nodes                         N,M     := NUM (Numbers or Operators)
# Numeric Value Nodes(!operators)       #n,#M   := NUM (where n,m \in \Q)
#
#
#

# <alphanumeric> ::= [a-zA-Z0-9_.-/]+

# main port of root node is "free" and unconnected
# connect two main ports and form reducible expression (redex)
# <Redex> ::= <Tree> "~" <Tree>

#net consists of a root tree and a (possibly empty) list of &-separated redexes
# <Net> ::= <Tree> ("&" <Redex>)*
# nets represent configurations/figures => only single free main port

#book consists of defined/named nets
# <Book> ::= ("@" <name> "=" <Net>)*

# aux - aux : wires explicit through variable nodes
# aux - main : wires implicit through tree structure => critical to efficient memory storage


# INSTRUCTION MECHANISMS - INTERACTIONS


#
#
#
#
#
#

voc.populate("F_ERA;;Cherry;Durian;Elderberry;Fig;Grape;Hawthorn")
statevars = [("statevar1", spa.SemanticPointer),
             ("statevar2", spa.SemanticPointer),
             ("statevar3", spa.SemanticPointer),
             ("statevar4", int),
             ("dummyin", spa.SemanticPointer),
             ("bananapass", spa.SemanticPointer)
            ]

table = {
        (voc["Apple"], voc["Banana"], None, 1): (voc["Banana"], voc["Apple"], StateVar("statevar1", "bananapass"), 0), 
        (voc["Banana"], voc["Apple"], None, 0): (voc["Apple"], voc["Banana"], InputVar("a", "dummyin"), 1)
        }

model = spa.Network()
with model:
    kws = """Tree-Node Keywords:
        *                   F_ERA(ser)
        @                   F_REF(erence)
        <Numeric>           F_NUM(eric)
        (                   F_LCON(structor)
        )                   F_RCON(structor)
        {                   F_LDUP(licator)
        {                   F_RDUP(licator)
        $                   F_OPE(rator)
        ?                   F_SWI(tch)
        <alphanumeric>      F_VAR(iable)
    """
