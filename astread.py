import libsbml as l
import sympy as sp
import sympy.physics.units as u
from sympy.functions.elementary.complexes import *
import AST_TYPES as DBG #dictionary

TIME_SYMBOL = sp.Dummy("t")
DELAY_SYMBOL = sp.Function(sp.Dummy())

def processLambda(ast):
    raise ValueError ("AST_LAMBDA not implemented")

def getConstantSymb (ast):
    match ast.getType():
        case l.AST_CONSTANT_FALSE:	
            return sp.false
        case l.AST_CONSTANT_TRUE:	
            return sp.true
        case l.AST_CONSTANT_E:
            return sp.exp(1)
        case l.AST_CONSTANT_PI:	
            return sp.pi
        case l.AST_NAME_AVOGADRO:
            return u.avogadro_number
        case l.AST_INTEGER:	
            return ast.getValue()
        case l.AST_REAL:
            return ast.getValue()
            #return ast.getReal()
        case l.AST_REAL_E:
            return ast.getValue()
            #return ast.getReal()
        case l.AST_RATIONAL:
            return ast.getValue()
            #return ast.getReal()
        case l.AST_NAME_TIME:
            return TIME_SYMBOL
        case l.AST_NAME:
            return sp.symbols(ast.getName())
        case _:
            raise ValueError("getConstantSymb(): type not implemented")



def getFunctionSymb (ast):
        match ast.getType():
            case l.AST_FUNCTION:
                if ast.isOperator() or ast.isNumber():
                    raise ValueError ("getFunctionSymb(): type is AST_FUNCTION but ast.isOperator() or ast.isNumber()")
                name = ast.getName()
                return sp.Function(name)
            case l.AST_FUNCTION_ABS:
                return abs
            case l.AST_FUNCTION_ARCCOS:
                return sp.acos
            case l.AST_FUNCTION_ARCCOSH:
                return sp.acosh
            case l.AST_FUNCTION_ARCCOT:
                return sp.acot
            case l.AST_FUNCTION_ARCCOTH:
                return sp.acoth
            case l.AST_FUNCTION_ARCCSC:
                return sp.acsc
            case l.AST_FUNCTION_ARCCSCH:
                return sp.acsch
            case l.AST_FUNCTION_ARCSEC:
                return sp.asec
            case l.AST_FUNCTION_ARCSECH:
                return sp.asech
            case l.AST_FUNCTION_ARCSIN:
                return sp.asin
            case l.AST_FUNCTION_ARCSINH:
                return sp.asinh
            case l.AST_FUNCTION_ARCTAN:	
                return sp.atan
            case l.AST_FUNCTION_ARCTANH:
                return sp.atanh
            case l.AST_FUNCTION_CEILING:
                return sp.ceiling
            case l.AST_FUNCTION_COS:
                return sp.cos
            case l.AST_FUNCTION_COSH:
                return sp.cosh
            case l.AST_FUNCTION_COT:
                return sp.cot
            case l.AST_FUNCTION_COTH:
                return sp.coth
            case l.AST_FUNCTION_CSC:
                return sp.csc
            case l.AST_FUNCTION_CSCH:
                return sp.csch
            case l.AST_FUNCTION_FLOOR:
                return sp.floor
            case l.AST_FUNCTION_EXP:
                return sp.exp
            case l.AST_FUNCTION_FACTORIAL:
                return sp.factorial
            case l.AST_FUNCTION_LN:
                return sp.ln
            case l.AST_FUNCTION_LOG:
                return sp.log
            case l.AST_FUNCTION_SEC:
                return sp.sec
            case l.AST_FUNCTION_SECH:
                return sp.sech
            case l.AST_FUNCTION_SIN:
                return sp.sin
            case l.AST_FUNCTION_SINH:
                return sp.sinh
            case l.AST_FUNCTION_TAN:
                return sp.tan
            case l.AST_FUNCTION_TANH:
                return sp.tanh
            case l.AST_FUNCTION_MAX:
               return sp.Max
            case l.AST_FUNCTION_MIN:
                return sp.Min
            case l.AST_FUNCTION_ROOT:
                return sp.root
            #polynomial quotients
            case l.AST_FUNCTION_QUOTIENT:
                return sp.quo
            case l.AST_FUNCTION_REM:
                return sp.rem


            case l.AST_FUNCTION_DELAY:	
                return DELAY_SYMBOL
            case l.AST_FUNCTION_PIECEWISE:	
                return sp.Piecewise
            case l.AST_FUNCTION_POWER:	
                if ast.getNumChildren() > 2:
                    raise ValueError("AST_FUNCTION_POWER with args {ast.getNumChildren()} > 2")
                return sp.Pow
            case l.AST_FUNCTION_RATE_OF:	
                raise ValueError("AST_FUNCTION_RATE_OF not implemented")
            case l.AST_LAMBDA:
                return processLambda(ast)
            case l.AST_NAME:
                if ast.isOperator() or ast.isNumber():
                    raise ValueError ("getFunctionSymb(): type is AST_NAME but ast.isOperator() or ast.isNumber()")
                name = ast.getName()
                #if ast.getNumChildren() == 0:
                #    return sp.symbols(name)
                #else:
                return sp.Function(name)

            case _:
                raise ValueError (f"getType() {DBG.types[ast.getType()]} not implemented")


def getLogicalSymb (ast):
    match ast.getType():
        case l.AST_LOGICAL_AND:
            return sp.And
        case l.AST_LOGICAL_NOT:
            return sp.Not
        case l.AST_LOGICAL_OR:
            return sp.Or
        case l.AST_LOGICAL_XOR:
            return sp.Xor
        case l.AST_LOGICAL_IMPLIES2:
            return sp.Implies
def getRelationalSymb (ast):
    match ast.getType():
        case l.AST_RELATIONAL_EQ:
            return sp.Eq
        case l.AST_RELATIONAL_GEQ:
            return sp.Ge
        case l.AST_RELATIONAL_GT:
            return sp.Gt
        case l.AST_RELATIONAL_LEQ:
            return sp.Le
        case l.AST_RELATIONAL_LT:
            return sp.Lt
        case l.AST_RELATIONAL_NEQ:
            return sp.Ne

def getOperatorSymb (ast):
    match ast.getType():
        case l.AST_PLUS:
            return sp.Add
        case l.AST_TIMES:
            return sp.Mul
        case l.AST_POWER:
            return sp.Pow
        case _:
            raise ValueError ("getOperatorSymb(): invalid type")

def getArguments (ast):
    children = []
    if ast.getType() == l.AST_FUNCTION_PIECEWISE:
        i=0
        while (i+1 < ast.getNumChildren()):
            x = ast.getChild(i)
            y = ast.getChild(i+1)
            children.append((astToExpr(x),astToExpr(y)))
            i+=2
        return children

    for i in range(ast.getNumChildren()):
        child = ast.getChild(i)
        expr = astToExpr(child)
        children.append(expr)
    return children

def processOperator (ast):
    match ast.getType():
        case l.AST_MINUS: 
            if ast.getNumChildren() > 2:
                raise ValueError (f"processOperator(): AST_MINUS with NumChildren() {ast.getNumChildren()}> 2")
            left = ast.getLeftChild()
            right = ast.getRightChild()
            if right == None:
                return - astToExpr(left)
            else:
                return astToExpr(left) - astToExpr(right)
        case l.AST_DIVIDE:	
            if ast.getNumChildren() > 2:
                raise ValueError (f"processOperator(): AST_DIVIDE with NumChildren() {ast.getNumChildren()}> 1")
            left = astToExpr(ast.getLeftChild())
            right = astToExpr(ast.getRightChild())
            return left / right
        case _:
            symb = getOperatorSymb(ast)
            children = getArguments(ast)
            return symb(*children)

def processCallable(ast):
    if ast.isLogical():
        symb = getLogicalSymb(ast)
    elif ast.isRelational():
        symb = getRelationalSymb(ast)
    else:
        symb = getFunctionSymb(ast)
    args = getArguments(ast)
    return symb(*args)


def astToExpr(ast):    
    #if ast.getType() == l.AST_ORIGINATES_IN_PACKAGE2:
    #    raise ValueError("AST_ORIGINATES_IN_PACKAGE2 not implemented")
    if ast.getType() == l.AST_UNKNOWN:
        raise ValueError("AST_UNKNOWN type")
    ast.canonicalize()
    if ast.isConstant() or ast.isNumber() or ast.getType() == l.AST_NAME_TIME:
        return getConstantSymb(ast)
    elif ast.isOperator():
        return processOperator(ast)
    elif(
         ast.isFunction() or 
         ast.getType() == l.AST_NAME or
         ast.isRelational() or
         ast.isLogical()
        ):
        return processCallable(ast)
    else:
        raise ValueError (f"astToExpr() type {DBG.types[ast.getType()]} not processed")

def parse_expr(expr):
    try:
        return astToExpr(expr) #, global_dict=global_dict, transformations=(auto_symbol,auto_number,factorial_notation))
    except TypeError as err:
        print ("parse error for expression: ", libsbml.formulaToL3String(expr))
        print (err)
        raise err
    except ValueError as err:
        print ("parse error for expression: ", libsbml.formulaToL3String(expr))
        print (err)
        raise err
    except _ as err:
        print ("parse error for expression: ", libsbml.formulaToL3String(expr))
        print (err)
        raise err
