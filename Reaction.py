

import sympy as sp

import libsbml
from Specie import *
from Parameter import *
import numbers

import sympy2jax

class Reaction(libsbml.Reaction):
    def __init__ (self,reaction, functions, symdict, assignmentrules):
        """
            Inputs:
                - the function list
                - the dictionary containing the unique symbols
                - the rules
                

        """
        super().__init__(reaction)
        self.localparameters = {}
        law = reaction.getKineticLaw()
        if None == law:
            raise ValueError(f"reaction {reaction.getId()} does not have a kinetic law")
        self.symbolic_formula = parse_expr(law.getMath())
        localsymbs = {}
        #local parameters (depending on the version of the SBML)
        for param in reaction.getKineticLaw().getListOfLocalParameters(): #newer versions
            parameter = Parameter(param)
            self.localparameters[param.getId()] = parameter
            localsymbs[sp.Function(parameter.getId())()] = parameter.symbol
        for param in reaction.getKineticLaw().getListOfParameters(): #old versions
            parameter = Parameter(param)
            self.localparameters[param.getId()] = parameter
            localsymbs[sp.Function(parameter.getId())()] = parameter.symbol
        if not isinstance(self.symbolic_formula, numbers.Number):
            self.symbolic_formula = self.symbolic_formula.subs(localsymbs)
        self.expandFunctions(functions)
        self.expandFunctions(assignmentrules)
        self.symbolic_formula = self.symbolic_formula.subs(symdict)

        try:
            self.kineticformula = sympy2jax.SymbolicModule([self.symbolic_formula])
        except KeyError as err:
            print ("Error: unable to convert a symbolic expression to jax. Are the operations differentiable ?")
            print ("Expression: ", err)
            print ("Non-differentiable expression will be handled in a future version.")
            raise

    def expandFunctions(self, functions):
        expression = self.symbolic_formula
        while True: #because there is no infinite recursions in python (nor do-until syntax) -_-
            recursive_call = False
            for func in expression.atoms(sp.Function):
                name = func.func.__name__
                if name in functions:
                    expression = functions[name].apply(expression)
                    recursive_call = True
            if recursive_call == False:
                break
        self.symbolic_formula = expression


