import libsbml
from astread import parse_expr
import sympy
import numbers



class Function(libsbml.FunctionDefinition):
    def __init__(self, function):
        super().__init__(function)
        body = function.getBody()
        args = []
        for i in range(function.getNumArguments()):
            args.append(function.getArgument(i).getName())

        #to avoid situations where variables have the same name as functions 
        #(i see you model 770)
        uniq_args = {sym:sympy.Function(sympy.Dummy())() for sym in args} 
        uniq_fun = {sympy.Function(sym)():val for sym,val in uniq_args.items()}

        self.args = uniq_args.values()
        self.name = function.getId()
        self.symbol = sympy.Function(self.name)(*self.args)


        self.expr = parse_expr(body)
        if not isinstance(self.expr, numbers.Number):
            self.expr = self.expr.subs(uniq_fun)
        print (self.symbol,"=",self.expr)

    def apply(self, expression):       
        for func in expression.atoms(sympy.Function):
            if func.func.__name__ == self.name:
                calledargs = func.args #the arguments on which the function is called
                symmap = dict(zip (self.args, calledargs))
                if not isinstance(self.expr, numbers.Number): #when the expression is not constant
                    newexpr = self.expr.subs(symmap) #substitute the variables in the definition with the arguments of the function call
                    expression = expression.subs(func, newexpr)
                else:
                    expression = expression.subs(func, self.expr)
        return expression       


