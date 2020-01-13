import numpy as np
import pandas as pd
import networkx as nx
import math
from scipy import stats
from SepMe.graph import add_node_attr

np.seterr(over="raise")


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


def neighbour_purity(graph, df, purity_type=["cp", "ce", "mv"]):
    # per datapoint results
    # we don't use networkx to store attributes in because it's slow and annoying
    classes = set(df["class"])
    number_of_classes = len(classes)
    if number_of_classes == 1:
        number_of_classes = 2

    cp = []
    ce = []
    mv_true = []
    mv_false = []
    neighbours = []
    nns = []

    for node in graph.nodes(data=True):
        neighbour_classes = list(df.loc[graph.neighbors(node[0]), "class"])
        amount_of_neighbours = len(neighbour_classes)
        neighbours.append(amount_of_neighbours)
        nns.append(neighbour_classes)

        self_class = node[1]["class"]
        count_of_self = neighbour_classes.count(self_class)

        if "cp" in purity_type:
            # node[1]['cp'] = count_of_self/len(neighbour_classes)

            if amount_of_neighbours == 0:
                # print(node)
                cp.append(1)
            else:
                cp.append(count_of_self / amount_of_neighbours)
                # print(count_of_self / amount_of_neighbours)

        if "ce" in purity_type:
            hi = 0

            if amount_of_neighbours == 0:
                ce.append(0)
            else:
                for c in classes:

                    if self_class == c:
                        qi = (count_of_self + 1) / (amount_of_neighbours + 1)
                    else:
                        qi = neighbour_classes.count(c) / (amount_of_neighbours + 1)

                    hi -= qi * math.log(qi, number_of_classes) if qi != 0 else 0

                # node[1]['ce'] = hi

                ce.append(hi)

        if "mv" in purity_type:

            if amount_of_neighbours == 0:
                mv_true.append(1.0)
                mv_false.append(1.0)
            else:
                vote = get_vote(neighbour_classes, True, self_class)
                if vote == self_class:
                    mv_true.append(1.0)
                else:
                    mv_true.append(0.0)

                vote = get_vote(neighbour_classes, False, self_class)
                if vote == self_class:
                    mv_false.append(1.0)
                else:
                    mv_false.append(0.0)

        if "wv" in purity_type:
            pass

    if len(neighbours) == len(df):
        df["mv_true"] = mv_true
        df["mv_false"] = mv_false
        df["neighbours"] = neighbours
        df["ce"] = ce
        df["cp"] = cp
        df["nns"] = nns

    # print(df.columns)
    # print('======')
    # print(df.head())
    # df.to_csv("heyhey.csv")
    return df


def total_neighbour_purity(df, graph, purity_type=["cp", "ce", "mv"]):
    df = neighbour_purity(graph, df, purity_type)
    stats = {}

    stats["cp_a"] = np.mean(df["cp"])
    nns = np.sum(df["neighbours"])
    if nns > 0:
        stats["ce_a"] = np.sum(df["ce"] * df["neighbours"]) / np.sum(df["neighbours"])
    else:
        stats["ce_a"] = -1

    stats["mv_a_true"] = np.mean(df["mv_true"])
    stats["mv_a_false"] = np.mean(df["mv_false"])

    for cc in set(df["class"]):
        stats["cp_{}".format(cc)] = np.mean(df.loc[df["class"] == cc, "cp"])

        nns = np.sum(df.loc[df["class"] == cc, "neighbours"])
        if nns > 0:
            stats["ce_{}".format(cc)] = np.sum(
                df.loc[df["class"] == cc, "ce"]
                * df.loc[df["class"] == cc, "neighbours"]
            ) / np.sum(df.loc[df["class"] == cc, "neighbours"])
        else:
            stats["ce_{}"] = -1
        stats["mv_{}_true".format(cc)] = np.mean(df.loc[df["class"] == cc, "mv_true"])
        stats["mv_{}_false".format(cc)] = np.mean(df.loc[df["class"] == cc, "mv_false"])

    return stats


def ltcc(graph, df):
    G = graph.copy()

    rem_edges = []
    for edge in G.edges():
        # print(edge)
        node_a = G.nodes(data=True)[edge[0]]["class"]
        node_b = G.nodes(data=True)[edge[1]]["class"]

        if node_a != node_b:
            rem_edges.append(edge)

    G.remove_edges_from(rem_edges)
    undir_graph = G.to_undirected()
    undir_graph = add_node_attr(undir_graph, df, False)
    a = np.array(
        [
            [list(undir_graph.subgraph(c).nodes(data=True))[0][1]["class"], len(c)]
            for c in nx.connected_components(undir_graph)
        ]
    )

    classes = set(df["class"])
    stats = {}

    stats["all"] = np.max(a, axis=0)[1] / np.sum(a, axis=0)[1]
    for c in classes:
        stats[c] = (
            np.max(a[a[:, 0] == c], axis=0)[1] / np.sum(a[a[:, 0] == c], axis=0)[1]
        )

    return stats


def get_amount_mixed(graph, class_name="class"):
    rem_edges = 0
    for edge in graph.edges():
        # print(edge)
        node_a = graph.nodes(data=True)[edge[0]][class_name]
        node_b = graph.nodes(data=True)[edge[1]][class_name]

        if node_a != node_b:
            rem_edges += 1
    return rem_edges


def mcec(graph, df, m, class_name="class"):

    nx.set_node_attributes(graph, df[[class_name]].to_dict("index"))
    mixed = []
    rem_edges = get_amount_mixed(graph)
    mixed.append(rem_edges)
    # print(rem_edges)

    for i in range(m):
        ddf = df.copy()
        ddf[class_name] = np.random.permutation(df[class_name])
        nx.set_node_attributes(graph, ddf[[class_name]].to_dict("index"))

        mixed.append(get_amount_mixed(graph, class_name))

    mixed = np.array(mixed)
    # print(mixed)
    # reset node attr just in case
    nx.set_node_attributes(graph, df[[class_name]].to_dict("index"))
    return len(mixed[mixed > rem_edges]) / m


def all_mcec(graph, df, m):
    class_labels = list(set(df["class"]))
    df_class = pd.get_dummies(df["class"])
    df_class.columns = ["class_{}".format(dd) for dd in df_class]
    df1 = pd.concat([df, df_class], axis=1)

    stats = {}
    # print(df1.columns)
    for cc in class_labels:
        stats[cc] = mcec(graph, df1, m, class_name="class_{}".format(cc))

    return stats
