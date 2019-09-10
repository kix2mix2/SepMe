import networkx as nx
from scipy.spatial import Delaunay, ConvexHull
from scipy.spatial.distance import euclidean
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph


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
        raise nx.NetworkXError('G and H must both be graphs or multigraphs.')
    R = nx.create_empty_copy(G)

    if set(G) != set(H):
        raise nx.NetworkXError("Node sets of graphs not equal")

    if G.is_multigraph():
        edges = G.edges(keys = True)
    else:
        edges = G.edges(data = True)
        # print(edges)
    for e in edges:
        if not H.has_edge(*e[:2]):
            # print(e)
            R.add_edge(*e[:2], weight = e[2]['weight'])
    return R


def add_node_attr(graph, df):
    # add node position and class as attributes
    for n in graph.nodes():
        graph.node[n]['pos'] = list(df.loc[n, ['x', 'y']])
        graph.node[n]['class'] = int(df.loc[n, ['class']])
    return graph


def get_delaunay(df, with_tri=False):
    graph = nx.Graph()
    tri = Delaunay(df[['x', 'y']])

    edges = set()
    for n in range(tri.nsimplex):
        edge = sorted([tri.simplices[n, 0], tri.simplices[n, 1]])
        edges.add((edge[0], edge[1],
                   euclidean((df.loc[tri.simplices[n, 0], ['x', 'y']]), (df.loc[tri.simplices[n, 1], ['x', 'y']]))))
        edge = sorted([tri.simplices[n, 0], tri.simplices[n, 2]])
        edges.add((edge[0], edge[1],
                   euclidean((df.loc[tri.simplices[n, 0], ['x', 'y']]), (df.loc[tri.simplices[n, 2], ['x', 'y']]))))
        edge = sorted([tri.simplices[n, 1], tri.simplices[n, 2]])
        edges.add((edge[0], edge[1],
                   euclidean((df.loc[tri.simplices[n, 1], ['x', 'y']]), (df.loc[tri.simplices[n, 2], ['x', 'y']]))))

    graph.add_weighted_edges_from(edges)

    if with_tri:
        tri.close()
        return graph, tri

    return graph


def get_convex_hull(df):
    graph = nx.Graph()
    hull = ConvexHull(df[['x', 'y']])

    weighted_edges = []
    for edge in hull.simplices:
        e = (edge[0], edge[1], euclidean(df.loc[edge[0], ['x', 'y']], df.loc[edge[1], ['x', 'y']]))
        weighted_edges.append(e)

    graph.add_weighted_edges_from(weighted_edges)
    return graph


# uses Delaunay graph
def get_mst(graph):
    return nx.minimum_spanning_tree(graph)


def get_knntree(df, n=2):
    X = df[['x', 'y']]
    A = kneighbors_graph(X, n, mode = 'distance', include_self = True)
    A.toarray()
    graph = nx.from_numpy_matrix(A.toarray())
    # nx.draw(graph, pointIDXY, node_size=25)
    return graph


def get_balltree(df, radius=30):
    X = df[['x', 'y']]
    A = radius_neighbors_graph(X, radius, mode = 'distance', include_self = True)
    A.toarray()
    graph = nx.from_numpy_matrix(A.toarray())
    # nx.draw(graph, pointIDXY, node_size=25)
    return graph


def get_rng(df, graph_del, graph_mst):
    # get all edge who are in DT nad not in EMST
    candidate_graph = attr_difference(graph_del, graph_mst)
    candidate_graph = add_node_attr(candidate_graph, df)

    remove_list = []
    for edge in candidate_graph.edges(data = True):
        edge_weight = edge[2]['weight']
        for possible_blocker in candidate_graph.nodes(data = True):
            pos = possible_blocker[1]['pos']
            dist_n0 = euclidean(pos, candidate_graph.nodes(data = True)[edge[0]]['pos'])
            dist_n1 = euclidean(pos, candidate_graph.nodes(data = True)[edge[1]]['pos'])
            if dist_n0 < edge_weight and dist_n1 < edge_weight:
                remove_list.append(edge)

    graph_rng = graph_del.copy()
    graph_rng.remove_edges_from(remove_list)

    print(len(graph_rng.edges(data = True)))
    print(len(graph_del.edges(data = True)))

    return graph_rng