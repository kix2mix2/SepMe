import pandas as pd
import numpy as np
import networkx as nx
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
from scipy.spatial.distance import euclidean, cdist, pdist, squareform
from scipy.spatial import Delaunay, ConvexHull
import bisect
from .workflow_utils import timeit
from profilehooks import profile


def attr_difference(G, H):
    """Returns a new graph that contains the edges with attributes that exist in G but not in H.

    The node sets of H and G must be the same.

    Parameters
    ----------
    G,H : graph
       A NetworkX graph.  G and H must have the same node sets.

    Returns
    -------
    D : A new graph with the same type as G.
    """
    # create new graph
    if not G.is_multigraph() == H.is_multigraph():
        raise nx.NetworkXError("G and H must both be graphs or multigraphs.")
    R = nx.create_empty_copy(G)

    if set(G) != set(H):
        raise nx.NetworkXError("Node sets of graphs not equal")

    if G.is_multigraph():
        edges = G.edges(keys=True)
    else:
        edges = G.edges(data=True)
        # print(edges)
    for e in edges:
        if not H.has_edge(*e[:2]):
            # print(e)
            R.add_edge(*e[:2], weight=e[2]["weight"])
    return R


def add_node_attr(graph, df, color=False):
    # add node position and class as attributes
    colors = []
    for n in graph.nodes():
        graph.node[n]["pos"] = list(df.loc[n, ["x", "y"]])
        graph.node[n]["class"] = int(df.loc[n, ["class"]])
        if color:
            if int(df.loc[n, ["class"]]):
                colors.append("blue")
            else:
                colors.append("red")

    if color:
        return (graph, colors)
    else:
        return graph


def get_delaunay(df, with_tri=False):
    graph = nx.Graph()
    tri = Delaunay(df[["x", "y"]])

    edges = set()
    for n in range(tri.nsimplex):
        edge = sorted([tri.simplices[n, 0], tri.simplices[n, 1]])
        edges.add(
            (
                edge[0],
                edge[1],
                euclidean(
                    (df.loc[tri.simplices[n, 0], ["x", "y"]]),
                    (df.loc[tri.simplices[n, 1], ["x", "y"]]),
                ),
            )
        )
        edge = sorted([tri.simplices[n, 0], tri.simplices[n, 2]])
        edges.add(
            (
                edge[0],
                edge[1],
                euclidean(
                    (df.loc[tri.simplices[n, 0], ["x", "y"]]),
                    (df.loc[tri.simplices[n, 2], ["x", "y"]]),
                ),
            )
        )
        edge = sorted([tri.simplices[n, 1], tri.simplices[n, 2]])
        edges.add(
            (
                edge[0],
                edge[1],
                euclidean(
                    (df.loc[tri.simplices[n, 1], ["x", "y"]]),
                    (df.loc[tri.simplices[n, 2], ["x", "y"]]),
                ),
            )
        )

    graph.add_weighted_edges_from(edges)

    if with_tri:
        tri.close()
        return graph, tri

    return graph


def get_convex_hull(df):
    graph = nx.Graph()
    hull = ConvexHull(df[["x", "y"]])

    weighted_edges = []
    for edge in hull.simplices:
        e = (
            edge[0],
            edge[1],
            euclidean(
                df.loc[edge[0], ["x", "y"]], df.loc[edge[1], ["x", "y"]]
            ),
        )
        weighted_edges.append(e)

    graph.add_weighted_edges_from(weighted_edges)
    return graph


# uses Delaunay graph


def get_mst(graph):
    return nx.minimum_spanning_tree(graph)


def get_knntree(df, n=1):
    X = df[["x", "y"]]
    A = kneighbors_graph(X, n + 1, mode="distance", include_self=True)
    A.toarray()
    graph = nx.from_numpy_matrix(A.toarray(), create_using=nx.DiGraph)
    # nx.draw(graph, pointIDXY, node_size=25)
    return graph


def get_balltree(df, radius=30):
    X = df[["x", "y"]]
    A = radius_neighbors_graph(X, radius, mode="distance", include_self=True)
    A.toarray()
    graph = nx.from_numpy_matrix(A.toarray())
    # nx.draw(graph, pointIDXY, node_size=25)
    return graph


@timeit
def get_rng(df, graph_del, graph_mst):
    # get all edge who are in DT nad not in EMST
    candidate_graph = attr_difference(graph_del, graph_mst)
    candidate_graph = add_node_attr(candidate_graph, df)

    remove_list = []
    for edge in candidate_graph.edges(data=True):
        edge_weight = edge[2]["weight"]
        for possible_blocker in candidate_graph.nodes(data=True):
            pos = possible_blocker[1]["pos"]
            dist_n0 = euclidean(
                pos, candidate_graph.nodes(data=True)[edge[0]]["pos"]
            )
            dist_n1 = euclidean(
                pos, candidate_graph.nodes(data=True)[edge[1]]["pos"]
            )
            if dist_n0 < edge_weight and dist_n1 < edge_weight:
                remove_list.append(edge)

    graph_rng = graph_del.copy()
    graph_rng.remove_edges_from(remove_list)

    return graph_rng


@timeit
def get_kncg(df, K=4):
    graph = get_knntree(df, 1)
    for node_a, row in df.iterrows():
        if node_a % 50:
            print(K)
        node_a_coord = list(row[:2])
        ncns = []
        ncn_coords = []

        for nn in graph.neighbors(node_a):
            ncns.append(nn)
            ncn_coords.append(list(df.loc[nn, ["x", "y"]]))

        # print(ncns)
        for k in range(2, K + 1):
            dist = -1
            chosen_one = 0
            for node_b, row_b in df.iterrows():
                if node_b in ncns + [node_a]:
                    continue
                else:
                    node_b_coord = list(row_b[:2])
                    centroid = np.mean(ncn_coords + [node_b_coord], axis=0)
                    # print(centroid)
                    cdist = euclidean(node_a_coord, centroid)

                    if dist == -1 or cdist < dist:
                        chosen_one = node_b
                        chosen_one_coord = node_b_coord
                        dist = cdist
            # print(dist)
            ncns.append(chosen_one)
            graph.add_edge(
                node_a,
                chosen_one,
                weight=euclidean(node_a_coord, chosen_one_coord),
            )
    return graph


@timeit
def get_gong(df, y=0):
    graph = nx.DiGraph()
    graph.add_nodes_from(df.iterrows())
    dists = pdist(df[["x", "y"]])
    dists = squareform(dists)
    y_dists = (1 - y) * dists

    for node_a, row_a in df.iterrows():
        if node_a % 50:
            print(y)
        node_a_coord = list(row_a[:2])  # O(dn)
        dist_idx = np.argsort(dists[node_a])  # O(nlog n)
        for node_b in dist_idx:
            if node_a == node_b:
                continue

            node_b_coord = list(df.loc[node_b, ["x", "y"]])

            d_i = y_dists[node_a][node_b]
            first_greater = bisect.bisect_left(dists[node_a][dist_idx], d_i)

            b_is_GONG = True

            for node_j in dist_idx[:first_greater]:
                if node_a == node_j:
                    continue

                d_j = y_dists[node_a][node_j]

                if d_j < d_i:
                    b_is_GONG = False
                    break  # node_j could be a GONG

            if b_is_GONG:
                graph.add_edge(node_a, node_b, weight=dists[node_a][node_b])
    return graph


def get_as(df, alpha=0.05):
    graph = get_delaunay(df)
    del_edges = []
    for edge in graph.edges(data=True):
        distance = edge[2]["weight"]
        if distance > 2 / alpha:
            del_edges.append(edge)
        else:
            pass

    graph.remove_edges_from(del_edges)
    return graph


# maybe don't use this
@timeit
def get_CBSG(df, beta=0, dists=None):
    graph = nx.Graph()
    graph.add_nodes_from(df.iterrows())

    if dists == None:
        dists = pdist(df[["x", "y"]])
        dists = squareform(dists)

    angleMax = np.pi * 0.5 * (1 + beta)

    for node_a, row_a in df[["x", "y"]].iterrows():
        for node_b, row_b in df[["x", "y"]].iterrows():
            if node_a == node_b:
                continue

            novois = 0
            for node_k, row_k in df[["x", "y"]].iterrows():
                if node_k == node_b or node_k == node_a:
                    continue

                ak = np.array(row_a - row_k)
                bk = np.array(row_b - row_k)

                try:
                    cosine_angle = np.dot(ak, bk) / (
                        dists[node_a, node_k] * dists[node_b, node_k]
                    )
                    if cosine_angle > 1 or cosine_angle < -1:
                        novois = 1
                        break

                    angle = np.arccos(cosine_angle)

                except:
                    novois = 1
                    angle = angleMax + 1
                    break

                # break
                if angle > angleMax:
                    novois = 1
                    break
            # break
            if novois == 0:
                graph.add_edge(node_a, node_b, weight=dists[node_a][node_b])
        # break
    return graph
