import nengo
import nengo_spa as spa
import numpy as np

from neural_dfa import DFA, InputVar, StateVar 

#dimentions and default vocabulary initialized here
d = 256 
voc = spa.Vocabulary(d)
# voc.populate


# Interaction Combinator system -> represented by interaction calculus system 
# seven types of agents (nodes) and variables -> both create tree 

#<Node> ::=
    # | "*"                       -- (ERA)ser
    # | "@" <alphanumeric>        -- (REF)erence
    # | <Numeric>                 -- (NUM)eric
    # | "(" <Tree> <Tree> ")"     -- (CON)structor
    # | "{" <Tree> <Tree> "}"     -- (DUP)licator
    # | "$(" <Tree> <Tree> ")"    -- (OPE)rator
    # | "?(" <Tree> <Tree> ")"    -- (SWI)tch
#<Tree> ::=
    # | <alphanumeric>          -- (VAR)iable
    # | <Node>        
