import pulp as solver
from pulp import *
import time
import gc
z = 0
solvers = [solver.CPLEX(timeLimit=30), solver.GLPK(), solver.GUROBI(), solver.PULP_CBC_CMD(), solver.COIN()]
solverUsado = 0

def solver_exact(g, init, final, is_first, is_last, name):

    var = [(i+','+str(t)) for i in g.edge for t in range(1,g.z)]
    var2 = [(str(i)+','+str(t)) for i in g.vertices for t in range(1,g.z)]
    X = solver.LpVariable.dicts("X",var,cat=solver.LpBinary)
    Y = solver.LpVariable.dicts("Y",var2,cat=solver.LpBinary)
    problem = solver.LpProblem("The best Cut" , solver.LpMinimize)
    if is_first:
        problem += (solver.lpSum(X.get(str(i.split(',')[0])+','+str(i.split(',')[1])+','+str(t))*g.mis[i.split(',')[0]][i.split(',')[1]] for i in g.edge for t in range(1,g.z)) + 
                    solver.lpSum(((g.pis[k.split(',')[0]][k.split(',')[1]]))-(
                                       (g.mis[k.split(',')[0]][k.split(',')[1]])) for k in g.edgeCuts)/2)+solver.lpSum(X.get(str(i.split(',')[0])+','+str(i.split(',')[1])+','+str(1))*g.initDes[int(i.split(',')[0])-1] for i in g.edge)
    else:
        problem += (solver.lpSum(X.get(str(i.split(',')[0])+','+str(i.split(',')[1])+','+str(t))*g.mis[i.split(',')[0]][i.split(',')[1]] for i in g.edge for t in range(1,g.z)) + 
                    solver.lpSum(((g.pis[k.split(',')[0]][k.split(',')[1]]))-(
                                       (g.mis[k.split(',')[0]][k.split(',')[1]])) for k in g.edgeCuts)/2)
    
    for t in range(1,g.z):
        problem += solver.lpSum(X.get(str(i.split(',')[0])+','+str(i.split(',')[1])+','+str(t)) for i in g.edge) <= 1
       
    for i in g.edgeCuts:
        problem += solver.lpSum(X.get(str(i.split(',')[0])+','+str(i.split(',')[1])+','+str(t))+X.get(str(i.split(',')[1])+','+str(i.split(',')[0])+','+str(t)) for t in range(1,g.z)) >= 1
    
    init_set = []
    for i in g.edge:
        if i.split(',')[0] == init:
            init_set.append(i)
    problem += solver.lpSum(X.get(str(i.split(',')[0])+','+str(i.split(',')[1])+','+str(1)) for i in init_set) == 1
    if not is_last:
        problem += solver.lpSum(Y.get(str(final)+','+str(t)) for t in range(1,g.z-1)) == 1
    
    for i in g.edge:
        for t in range(1,(g.z)-1):
            problem += (solver.lpSum(X.get(str(k)+','+str(i.split(',')[1])+','+str(t)) for k in g.arrive(i.split(',')[1])) - (solver.lpSum(X.get(str(i.split(',')[1])+','+str(j)+','+str(t+(1)))for j in g.leave(i.split(',')[1])))-Y.get(str(i.split(',')[1])+','+str(t))) == 0
    timeIn = time.time()
    #sCplex = solver.CPLEX()
    #solver.GLPK()
    st = problem.solve(solvers[solverUsado])

    values = []
    for k in problem.variables():
        if(solver.value(k) > 0):
            if(k.name.split('_')[0] == 'X'):
                values.append(k)
    valuesOr = [None for i in range(len(values))]
        
    for i in values:
        ind = int(i.name.split(',')[2])
        cut = i.name.split(',')[0].split('_')[1]+','+i.name.split(',')[1]
        valuesOr[ind-1] = cut

    #g.plotSoluation(valuesOr,nome.replace(".txt",""))
    #g.plotCuts(nome.replace(".txt",""))
    #g.plotCor(nome.replace(".txt",""))
    #g.plotDesloc(valuesOr,nome.replace(".txt",""))
    fo = solver.value(problem.objective)
    del problem
    del X
    del Y
    del g
    gc.collect()
    return fo,valuesOr
