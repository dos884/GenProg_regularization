
import operator
import math
import random

import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

import matplotlib as mpl


# Define new functions
def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1
    except OverflowError:
        return 1


def safePow(a,b):
    try:
        return (a**b).real
    except ZeroDivisionError:
        return 1
    except OverflowError:
        return 1

pset = gp.PrimitiveSet("MAIN", 1)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.neg, 1)
pset.addTerminal(2)
pset.addPrimitive(safePow, 2)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.sin, 1)
# pset.addEphemeralConstant("rand101", lambda: random.randint(-1, 1))
# pset.renameArguments(ARG0='x')

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


def targetFunc(x):
    #  the real function : x**4 + x**3 + x**2 + x
    #return x ** 4 - x ** 3 - x ** 2 - x
    #return x**2 +x**3 -x
    return x**4 + x**3 + x**2 + x

def empiricalLoss(func,points):
    sqerrors = ((func(x[0]) - x[1]) ** 2 for x in points)
    loss= (math.fsum(sqerrors) / len(points) )
    return loss

def empiricalFuncDiff(fun1,fun2,points):
    sqerrors = [math.fabs(fun1(x) - fun2(x)) for x in points]
    loss = (math.fsum(sqerrors) / len(points))
    return loss


OCCAM_PARAM = 1.0 / 10
BIG_NUM=6*10**64

import re
def individualLength(individual):
    st = individual.__str__().replace('(', ';')
    A = re.split(";|,", st)
    print(A)
    return len(A)

def evalSymbReg(individual, points):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    # Evaluate the mean squared error between the expression
    loss = empiricalFuncDiff(func, targetFunc, points) + (OCCAM_PARAM*len(individual.__str__()))
    if loss<0.00000001:
        loss = empiricalFuncDiff(func, targetFunc, points)
        L=individualLength(individual)
        a=0

    elif loss>BIG_NUM:
        loss = empiricalFuncDiff(func, targetFunc, points)
        st = individual.__str__()

    return loss,


toolbox.register("evaluate", evalSymbReg, points=[x / 10. for x in range(-10, 10)])
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))


def main():
    random.seed(318)

    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 150, stats=mstats,
                                   halloffame=hof, verbose=True)
    # print log
    print(hof.items[0].__str__())
    return pop, log, hof


if __name__ == "__main__":
    main()
