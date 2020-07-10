from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import Graph
import Exact
import random
import os
import signal
import gc


def greedy_min_path(min_bet_graph_values, vis):
    for i in tqdm(range(len(new_graphs)-1),desc="Greddy min path"):
        index = -1
        value = float('inf')
        last = vis[-1]
        for j in range(len(min_bet_graph_values)):
            if j not in vis:
                if min_bet_graph_values[j][last] < value:
                    value = min_bet_graph_values[j][last]
                    index = j
        vis.append(index)
    return vis

def shortest_path_clusters(new_graphs_v):
    min_bet_graph = [[] for i in range(len(new_graphs_v))]
    min_bet_graph_values = [[] for i in range(len(new_graphs_v))]
    for i in tqdm(range(len(new_graphs_v)),desc="shortest path between clusters"):
        for j in range(len(new_graphs_v)):
            v_min = float('inf')
            a_min = None
            for k in range(len(new_graphs_v[i])):
                for q in range(len(new_graphs_v[j])):
                    if i != j:
                        if new_graphs_v[i][k] == new_graphs_v[j][q]:
                            v_min = .0
                            a_min = new_graphs_v[i][k]+','+new_graphs_v[j][q]
                        elif g.distance[new_graphs_v[i][k]][new_graphs_v[j][q]] < v_min:
                            v_min = g.distance[new_graphs_v[i][k]][new_graphs_v[j][q]]
                            a_min = new_graphs_v[i][k]+','+new_graphs_v[j][q]
            min_bet_graph[i].append(a_min)
            min_bet_graph_values[i].append(v_min)
    return min_bet_graph, min_bet_graph_values

def shortest_clusters_origin(new_graphs_v, g):
    m = float('inf')
    index = -1
    for i in range(len(new_graphs_v)):
        for j in new_graphs_v[i]:
            ponto = g.points[int(j)-1]
            if (ponto[0]**2 + ponto[1]**2)**1/2 < m:
                m = (ponto[0]**2 + ponto[1]**2)**1/2
                index = i
    return index
def signal_handler(signum, frame):
    raise Exception("Timed out!")
    
problems_packing = ["Instances/packing/"+i for i in os.listdir("Instances/packing/")]
problems_sep = ["Instances/separated/"+i for i in os.listdir("Instances/separated/")]
problems = problems_packing + problems_sep
problems = problems[:4]
for ptk in problems:
    g = Graph.Graph(16.67,400,5)
    g.initProblem(ptk)
    g.z = len(g.edgeCuts)*2

    list_fo = []
    list_k = []
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(7200)
    try:
        for n_grupos in range(len(g.edgeCuts),2,-1):
            signal.signal(signal.SIGALRM, signal_handler)
            signal.alarm(600)
            try:    
                
                points = []
                
                mapa = dict()
                for i in g.edgeCuts:
                    init = int(i.split(",")[0])
                    final = int(i.split(",")[1])
                    p1 = g.points[init-1]
                    p2 = g.points[final-1]
                    pmx = (p1[0]+p2[0])/2
                    pmy = (p1[1]+p2[1])/2
                    mapa.update({i:[pmx,pmy]})
                
                n_clusters = n_grupos
                points = list(mapa.values())
                edges = list(mapa.keys())
                points = np.array(points)
                
                kmeans = KMeans(n_clusters=n_clusters, random_state=2222).fit(points)
                
                
                
                cores = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(100)]
                
                c = kmeans.predict(points)
                
                    
                index = 0
                new_graphs = [[] for i in range(n_clusters)]
                for i in range(len(c)):
                    new_graphs[c[i]].append(edges[index])
                    index += 1
                
                graph_min = [[] for i in range(n_clusters)]
                new_graphs_v = [[] for i in range(n_clusters)]
                
                for i in range(len(new_graphs)):
                    for j in new_graphs[i]:
                        if not j.split(',')[0] in new_graphs_v[i]:
                            new_graphs_v[i].append(j.split(',')[0])
                        if not j.split(',')[1] in new_graphs_v[i]:
                            new_graphs_v[i].append(j.split(',')[1])
                
                
                min_bet_graph, min_bet_graph_values = shortest_path_clusters(new_graphs_v)
                
                index = shortest_clusters_origin(new_graphs_v, g)
                
                vis = greedy_min_path(min_bet_graph_values, [index])
                min_bet_graph_n = [min_bet_graph[i] for i in vis]
                new_graphs_v_n = [new_graphs_v[i] for i in vis]
                new_graphs_ord = [new_graphs[i] for i in vis]
                min_bet_graph_values_n = [min_bet_graph_values[i] for i in vis]
                new_graphs = new_graphs_ord
                min_bet_graph = min_bet_graph_n
                new_graphs_v = new_graphs_v_n
                min_bet_graph_values = min_bet_graph_values_n
                
                min_bet_graph, min_bet_graph_values = shortest_path_clusters(new_graphs_v)
                
                
                obj_graphs = []
                for i in tqdm(range(len(new_graphs)),desc="Initializing objects clusters"):
                    graph = Graph.Graph(400,16.67,5)
                    graph.vertices = [i for i in range(1,len(new_graphs_v[i])+1)]
                    edgesC = []
                    for j in range(len(new_graphs[i])):
                        init = new_graphs[i][j].split(",")[0]
                        final = new_graphs[i][j].split(",")[1]
                        ed = str(new_graphs_v[i].index(init)+1) + ',' + str(new_graphs_v[i].index(final)+1)
                        edgesC.append(ed)
                    graph.edgeCuts = edgesC
                    for j in range(len(new_graphs_v[i])):
                        graph.points.append(g.points[int(new_graphs_v[i][j])-1])
                    for j in range(1,len(graph.points)+1):
                        for k in range(1,len(graph.points)+1):
                            if j != k:
                                graph.edge.append(str(j)+','+str(k))
                    graph.calculateDistances()
                    graph.z = len(graph.edgeCuts)*2
                    obj_graphs.append(graph)
                
                
                
                
                finals = ['' for i in range(len(min_bet_graph))]
                inits = ['' for i in range(len(min_bet_graph))]
                finals[-1] = str(new_graphs_v[-1].index(min_bet_graph[-1][0].split(',')[0])+1)
                for i in range(len(min_bet_graph)-1):
                    finals[i] = str(new_graphs_v[i].index(min_bet_graph[i][i+1].split(',')[0])+1)
                
                for i in range(1,len(min_bet_graph)):
                    inits[i] = str(new_graphs_v[i].index(min_bet_graph[i-1][i].split(',')[1])+1)
                inits[0] = str(new_graphs_v[0].index(min_bet_graph[-1][0].split(',')[1])+1)
                
                is_first = [False for i in range(len(obj_graphs))]
                is_last = [False for i in range(len(obj_graphs))]
                is_first[0] = True
                is_last[-1] = True
                
                fo = 0
                cluster_sol = []
                for i in tqdm(range(len(obj_graphs)),desc="Solving the clusters"):
                    v, sol = Exact.solver_exact(obj_graphs[i],inits[i],finals[i],is_first[i],is_last[i],"cluster"+str(i+1))
                    fo += v
                    cluster_sol.append(sol)
                count = 0
                vis = []
                #for i in range(len(cluster_sol)):
                 #   for j in range(len(cluster_sol[i])):
                  #      if cluster_sol[i][j] != None:
                   #         s = int(cluster_sol[i][j].split(',')[0])
                    #        f = int(cluster_sol[i][j].split(',')[1])
                     #       plt.quiver(obj_graphs[i].points[int(s)-1][0],obj_graphs[i].points[int(s)-1][1],obj_graphs[i].points[int(f)-1][0]-obj_graphs[i].points[int(s)-1][0],obj_graphs[i].points[int(f)-1][1]-obj_graphs[i].points[int(s)-1][1], scale_units='xy', angles='xy', scale=1,color=cores[i])
                      #      plt.text((obj_graphs[i].points[s-1][0]+obj_graphs[i].points[f-1][0])/2,(obj_graphs[i].points[s-1][1]+obj_graphs[i].points[f-1][1])/2,str(count)+" C: "+str(i))
                       #     count += 1
                
                for i in range(len(min_bet_graph_values)-1):
                    fo += min_bet_graph_values[i][i+1]/400
                list_fo.append(fo)
                list_k.append(n_grupos)
                del kmeans
                del mapa
                del inits
                del finals
                del c
                del min_bet_graph
                gc.collect()
                #print("Final FO value ",fo)
                #plt.title("Time Required: {:.2f}".format(fo))
                #plt.savefig("Images/"+ptk+".jpg",dpi=300)
                #plt.close()    
            except Exception as e:
                print(e)
                list_fo.append(float('inf'))
                list_k.append(n_grupos)
        save_str = "problem: "+ptk+" FO: "+str(min(list_fo)) + " k_value: "+str(list_k[list_fo.index(min(list_fo))])
        tempos = open("tempos.txt","a+")
        tempos.writelines(save_str+"\n")
        tempos.close()
    except Exception as e:
        print(e)
        try:
            save_str = "problem" + ptk + "FO: "+str(min(list_fo)) + "k_value: "+str(list_k[list_fo.index(min(list_fo))])
            tempos = open("tempos.txt","a+")
            tempos.writelines(save_str+"\n")
            tempos.close()
        except:
            pass
        pass
