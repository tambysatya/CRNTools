import sympy

import libsbml

class Parameter(libsbml.Parameter):
    def __init__ (self, parameter, name=None):
        super().__init__(parameter)
        if name != None:
            self.symbol = sympy.symbol(name)
        else:
            self.symbol = sympy.Dummy()
        
        self.val = parameter.getValue()

    def label(self, idx):
        self.symbol = sympy.symbol("k"+str(idx))



