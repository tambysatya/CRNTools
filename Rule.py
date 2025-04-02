

import libsbml

from astread import *

import sympy as sp



class Rule (libsbml.Rule):
    def __init__(self, rule):
        super().__init__(rule)
        self.formula = parse_expr(rule.getMath())

    def apply(self, expression):
        print ("applying rule: ", self.getVariable())
        var = self.getVariable()
        sym = sp.Function(var)()
        expression = expression.subs(sym, self.formula)
        return expression
