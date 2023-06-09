import numpy as np
import networkx as nx
import itertools
import matplotlib.pyplot as plt
import homophily_multi_attr_viz as homomul_viz
from collections import defaultdict


def marginal_matrix(g: nx.Graph, plot=True) -> np.ndarray:
    number_of_nodes_per_group = defaultdict(int)
    for n in g.nodes:
        attr = g.nodes[n]["attr"]
        number_of_nodes_per_group[attr] += 1

    for x1, x2 in itertools.product(range(2), repeat=2):
        number_of_nodes_per_group[(x1, x2)] += 0

    values = [number_of_nodes_per_group[k]
              for k in sorted(number_of_nodes_per_group.keys())]
    marginal_matrix = np.array(values).reshape(2, 2)
    marginal_matrix = marginal_matrix / marginal_matrix.sum()

    if plot:
        plt.figure(figsize=(4, 4))
        ax = plt.subplot(1, 1, 1)
        homomul_viz.fig_colored_matrix(
            marginal_matrix,
            ax=ax,
            xticks=range(2),
            yticks=range(2),
            show_colorbar=False,
            figsize=None,
            vmin=0)
        plt.show()

    return marginal_matrix


def homophily_matrix(g: nx.Graph, plot=True) -> np.ndarray:
    number_of_edges = defaultdict(int)
    number_of_nodes_per_group = defaultdict(int)

    for e in g.edges:
        n1 = g.nodes[e[0]]["attr"]
        n2 = g.nodes[e[1]]["attr"]
        number_of_edges[(n1, n2)] += 1
        number_of_edges[(n2, n1)] += 1

    for n in g.nodes:
        attr = g.nodes[n]["attr"]
        number_of_nodes_per_group[attr] += 1

    for x1, x2, x3, x4 in itertools.product(range(2), repeat=4):
        n1 = (x1, x2)
        n2 = (x3, x4)
        number_of_edges[(n1, n2)] += 0

        possible_edges = \
            number_of_nodes_per_group[n1] * number_of_nodes_per_group[n2]
        number_of_edges[(n1, n2)] = number_of_edges[(n1, n2)] / possible_edges

    values = [number_of_edges[k] for k in sorted(number_of_edges.keys())]
    number_of_edges_matrix = np.array(values).reshape(4, 4)

    number_of_edges_matrix = number_of_edges_matrix /\
        number_of_edges_matrix.max()

    if plot:
        plt.figure(figsize=(4, 4))
        ax = plt.subplot(1, 1, 1)
        homomul_viz.fig_colored_matrix(
            number_of_edges_matrix,
            ax=ax,
            xticks=itertools.product(range(2), repeat=2),
            yticks=itertools.product(range(2), repeat=2),
            show_colorbar=False,
            figsize=None,
            vmin=0)
        plt.show()

    return number_of_edges_matrix
