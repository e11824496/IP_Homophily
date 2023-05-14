import networkx as nx
import numpy as np


def complex_spread(g: nx.Graph,
                   initial_seed: list,
                   complex_threshold: float = 0.18):
    nx.set_node_attributes(g, False, 'contagion')

    for n in initial_seed:
        g.nodes[n]['contagion'] = True

    num_changes = 1

    while num_changes > 0:
        num_changes = 0
        for n in g.nodes():
            if g.nodes[n]['contagion']:
                continue

            neighbours = [to for frm, to in g.edges(n)]
            if len(neighbours) == 0:
                continue

            p = np.average([g.nodes[n]['contagion'] for n in neighbours])
            if p >= complex_threshold:
                g.nodes[n]['contagion'] = True
                num_changes += 1

    return g


def simple_spread(g: nx.Graph, initial_seed: list):
    nx.set_node_attributes(g, False, 'contagion')

    for n in initial_seed:
        g.nodes[n]['contagion'] = True

    for _ in range(10):
        for n in g.nodes():
            neighbours_infected = [g.nodes[to]['contagion']
                                   for frm, to in g.edges(n)]
            g.nodes[n]['contagion'] = any(neighbours_infected)

    return g


def fraction_infected(g: nx.Graph) -> float:
    return np.average([c for n, c in g.nodes(data='contagion')])
