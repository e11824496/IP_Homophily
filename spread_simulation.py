import networkx as nx
import numpy as np
import network_generation as homomul
from tqdm.notebook import tqdm
import homophily_multi_attr_viz as viz


def complex_spread(g: nx.Graph,
                   initial_seed: list,
                   complex_threshold: float = 0.16):
    nx.set_node_attributes(g, False, 'contagion')

    for n in initial_seed:
        g.nodes[n]['contagion'] = True

    iterations = 0
    num_changes = 1

    while num_changes > 0:
        iterations += 1
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

    return g, iterations


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


def fraction_infected(g: nx.Graph, result_filter=lambda x: x.nodes) -> float:
    filterd_nodes = result_filter(g)
    if len(filterd_nodes) == 0:
        return 0
    return np.average([g.nodes[n]['contagion'] for n in filterd_nodes])


def batch_simulate(default_model_setting: dict,
                   experiment_settings: dict) -> list[float]:

    n_networks = experiment_settings['n_networks']
    n_initial_seeds = experiment_settings['n_initial_seeds']

    initial_seed_filter = default_model_setting['initial_seed_filter']
    result_filter = default_model_setting.get(
        'result_filter', lambda x: x.nodes)
    complex_threshold = default_model_setting['complex_threshold']

    network_generation = homomul.am_v2
    if 'model_type' in default_model_setting:
        m = default_model_setting['model_type']
        network_generation = homomul.get_network_generator(m)

    results = []
    iteration_results = []

    for _ in range(n_networks):
        g = network_generation(**default_model_setting)

        if g is None:
            return np.NAN, np.NAN, np.NAN

        filterd_nodes = initial_seed_filter(g)
        n = min(n_initial_seeds, len(filterd_nodes))
        initial_seeds = np.random.choice(filterd_nodes, n, replace=False)

        for initial in initial_seeds:
            initial = [initial] + [x for x in g.neighbors(initial)]
            g, iterations = complex_spread(g, initial, complex_threshold)
            results.append(fraction_infected(g, result_filter))
            iteration_results.append(iterations)

    average_spread = np.average(results)
    average_global_cascades = np.average([x > 0.9 for x in results])
    average_iter_global_cascades = np.average(
        [i for r, i in zip(results, iteration_results) if r > 0.9])

    return (average_spread,
            average_global_cascades,
            average_iter_global_cascades)


def setting_simulate(v1_key, v1_settings, v2_key, v2_settings,
                     default_model_setting: dict, experiment_settings: dict,
                     visualize_results: bool = True,
                     label1=None, label2=None):

    results_average = np.zeros((v1_settings.size, v2_settings.size))
    results_global_spread = results_average.copy()
    results_iter = results_average.copy()

    progress_bar = tqdm(total=v1_settings.size * v2_settings.size)

    if label1 is None:
        label1 = v1_key
    if label2 is None:
        label2 = v2_key

    for i, v1 in enumerate(v1_settings):
        for j, v2 in enumerate(v2_settings):

            default_model_setting[v1_key] = v1
            default_model_setting[v2_key] = v2

            r_average, r_global_spread, r_iter = batch_simulate(
                default_model_setting,
                experiment_settings)
            tqdm.write(f'{label1}: {v1:.2f} / {label2}: {v2:0.2f} ' +
                       f'=> avg = {r_average:.2f}; ' +
                       f'global = {r_global_spread:0.2f}; ' +
                       f'avg iter = {r_iter:.1f}')

            results_average[i, j] = r_average
            results_global_spread[i, j] = r_global_spread
            results_iter[i, j] = r_iter

            progress_bar.update()

    progress_bar.close()

    if visualize_results:
        viz.fig_2attr_heatmap(label1, v1_settings, label2, v2_settings,
                              results_average, title='Average Spread')
        viz.fig_2attr_heatmap(label1, v1_settings, label2, v2_settings,
                              results_global_spread, title='Global Spread')
        viz.fig_2attr_heatmap(label1, v1_settings, label2, v2_settings,
                              results_iter, title='Average Iterations',
                              clim=None)

    return results_average, results_global_spread, results_iter
