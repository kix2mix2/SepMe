import numpy as np
import networkx as nx
import math
from scipy import stats
import random
import seaborn as sns
from SepMe.utils.graph_neighbourhoods import add_node_attr, attr_difference
from .workflow_utils import timeit

def get_vote(neighbours, pessimism=True, target_class=1):
    if len(neighbours) == 1:
        return neighbours[0]

    first_mode = stats.mode(neighbours)

    nn = neighbours.copy()
    nn.remove(first_mode.mode[0])  # this removes the first occurence of the mode
    second_mode = stats.mode(nn)

    if first_mode.count[0] == second_mode.count[0]:
        # this means there's a tie and there are two modes
        # if the target class is either of these modes, watch out for pessimism
        if target_class == first_mode.mode[0]:
            return second_mode.mode[0] if pessimism else target_class

        if target_class == second_mode.mode[0]:
            return first_mode.mode[0] if pessimism else target_class

    # if there isn't a tie, or target class is not in tied modes
    return first_mode.mode[0]



def neighbour_purity(graph, df, purity_type=['cp', 'ce', 'mv'], pessimism=True):
    # per datapoint results
    # we don't use networkx to store attributes in because it's slow and annoying
    node_scores = []

    classes = set(df['class'])
    number_of_classes = len(classes)
    if number_of_classes == 1:
        number_of_classes = 2

    cp = []
    ce = []
    mv = []
    wv = []

    for node in graph.nodes(data = True):
        neighbour_classes = list(df.loc[graph.neighbors(node[0]), 'class'])
        amount_of_neighbours = len(neighbour_classes)



        self_class = node[1]['class']
        count_of_self = neighbour_classes.count(self_class)

        if 'cp' in purity_type:
            # node[1]['cp'] = count_of_self/len(neighbour_classes)

            if amount_of_neighbours == 0:
                # print(node)
                cp.append(node[1]['class'])
            else:
                cp.append(count_of_self / amount_of_neighbours)

        if 'ce' in purity_type:
            hi = 0

            if amount_of_neighbours == 0:
                ce.append(node[1]['class'])
            else:
                for c in classes:

                    if self_class == c:
                        qi = (count_of_self + 1) / (amount_of_neighbours + 1)
                    else:
                        qi = neighbour_classes.count(c) / (amount_of_neighbours + 1)


                    hi -= (qi * math.log(qi, number_of_classes) if qi != 0 else 0)

            # node[1]['ce'] = hi


                ce.append(hi)

        if 'mv' in purity_type:

            if amount_of_neighbours == 0:
                mv.append(node[1]['class'])
            else:
                vote = get_vote(neighbour_classes, pessimism, self_class)
                mv.append(vote)


        if 'wv' in purity_type:
            pass

    if len(cp) == len(df):
        df['cp'] = cp
    if len(ce) == len(df):
        df['ce'] = ce
    if len(mv) == len(df):
        df['mv'] = mv
    if len(wv) == len(df):
        df['wv'] = wv

    return df



def total_neighbour_purity(df, graph, class_name=None, purity_type=['cp', 'ce', 'mv'], pessimism=False):
    df = neighbour_purity(graph, df, purity_type, pessimism)
    stats = {}
    if class_name != None:
        for pp in purity_type:
            stats[pp] = np.mean(df[df['class'] == class_name][pp])

    else:
        for pp in purity_type:
            stats[pp] = np.mean(df.loc[:, pp])

    return stats



def ltcc(graph, df):
    rem_edges = []
    for edge in graph.edges():
        # print(edge)
        node_a = graph.nodes(data = True)[edge[0]]['class']
        node_b = graph.nodes(data = True)[edge[1]]['class']

        if node_a != node_b:
            rem_edges.append(edge)

    graph.remove_edges_from(rem_edges)
    undir_graph = graph.to_undirected()
    undir_graph = add_node_attr(undir_graph, df, False)
    a = np.array([[list(undir_graph.subgraph(c).nodes(data = True))[0][1]['class'], len(c)] for c in
                  nx.connected_components(undir_graph)])

    stats = {c: np.max(a[a[:, 0] == c], axis = 0)[1] / np.sum(a[a[:, 0] == c], axis = 0)[1] for c in set(df['class'])}

    return stats



def mcec(graph, df, m):
    mixed = []

    rem_edges = 0
    for edge in graph.edges():
        # print(edge)
        node_a = graph.nodes(data = True)[edge[0]]['class']
        node_b = graph.nodes(data = True)[edge[1]]['class']

        if node_a != node_b:
            rem_edges += 1

    # mixed.append(rem_edges)

    # print('Mixed Edges: ' + str(rem_edges))
    classes = list(set(df['class']))
    # check distribution of mixed edges
    for i in range(m):
        j = 0
        for edge in graph.edges():
            # print(edge)
            node_a = random.choice(classes)
            node_b = random.choice(classes)

            if node_a != node_b:
                j += 1
        mixed.append(j)

    mixed = np.array(mixed)
    return len(mixed[mixed > rem_edges]) / m
