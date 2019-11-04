import networkx as nx
from .workflow_utils import timeit
from .graph_neighbourhoods import *
from .graph_purity import *
from SepMe import logger
import mlflow
import os
import pickle

@timeit
def calculate_graphs(graph_path, df, config):
    # ['del', 'ch', 'mst', 'knn', 'kncg', 'gong', 'rng', 'bt', 'as']
    # checks if graphs for this file were created.
    # does not check if all requested graphs are in there.
    if os.path.exists(graph_path):
        logger.info('Graph file already exists. Reloading...')
        with open(graph_path, 'rb') as handle:
            return pickle.load(handle)

    else:
        logger.info('Calculating Neighbourhood graphs...')

        graph_types = {}

        if 'del' in config:
            graph_types['del'] = add_node_attr(get_delaunay(df), df)

        if 'mst' in config:
            if graph_types['del'] is not None:
                graph_types['mst'] = get_mst(graph_types['del'])
            else:
                graph_types['del'] = get_delaunay(df)
                graph_types['mst'] = add_node_attr(get_mst(graph_types['del']), df)

        if 'knn' in config:
            graph_types['knn'] = {}
            for param in config['knn']:

                graph_types['knn'][param] = add_node_attr(get_knntree(df, param),df)

        if 'kncg' in config:
            graph_types['kncg'] = {}
            for param in config['kncg']:
                logger.info('kncg-param:' + str(param))
                graph_types['kncg'][param] = add_node_attr(get_kncg(df, param),df)

        if 'gong' in config:
            graph_types['gong'] = {}
            for param in config['gong']:
                logger.info('gong-param:' + str(param))
                graph_types['gong'][param] = add_node_attr(get_gong(df, param),df)

        if 'rng' in config:
            if graph_types['del'] is None:
                graph_types['del'] = get_delaunay(df)
            if graph_types['mst'] is None:
                graph_types['mst'] = graph_types['mst'] = get_mst(graph_types['del'])
            graph_types['rng'] = {}
            graph_types['rng'] = add_node_attr(get_rng(df, graph_types['del'], graph_types['mst']),df)

        if 'bt' in config:
            graph_types['bt'] = {}
            for param in config['bt']:
                graph_types['bt'][param] = add_node_attr(get_balltree(df, param),df)

        if 'as' in config:
            graph_types['as'] = {}
            for param in config['as']:
                graph_types['as'][param] = add_node_attr(get_as(df, param),df)

        if 'cbsg' in config:
            graph_types['cbsg'] = {}
            for param in config['cbsg']:
                graph_types['cbsg'][param] = add_node_attr(get_CBSG(df, param),df)

        with open(graph_path, 'wb') as handle:
            pickle.dump(graph_types, handle, protocol = pickle.HIGHEST_PROTOCOL)

        return graph_types


def calculate_purities(df, graph, purities):

    purity_dict = {}

    if 'ltcc' in purities['type']:
        purity_dict['ltcc'] = ltcc(graph, df)
    if 'mcec' in purities['type']:
        purity_dict['mcec'] = mcec(graph, df, 100)

    neighbour_purity_list = list(set(['cp', 'ce', 'mv']).intersection(set(purities['type'])))

    # print(total_neighbour_purity(df, graph, purities['class'], neighbour_purity_list, purities['pessimism']))
    for pessimism in purities['pessimism']:
        purity_dict.update(total_neighbour_purity(df, graph, None, neighbour_purity_list, pessimism))

    return purity_dict
