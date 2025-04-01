

import libsbml

from astread import *



class Rule (libsbml.Rule):
    def __init__(self, rule):
        super().__init__(rule)
        self.formula = parse_expr(rule.getMath())
