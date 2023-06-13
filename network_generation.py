import numpy as np
import itertools
import networkx as nx
import math


def interaction_all_normalization(h_matrix_lst: np.ndarray) -> float:
    return 1/np.prod(np.max(h_matrix_lst, axis=(1, 2)))


def interaction_all(h_vec: list, normalization: float) -> bool:
    return normalization * np.prod(h_vec) >= np.random.random()


#######################################
# Tunable consolidation for two binary populations
#######################################

def two_bin_comp_pop_frac_tnsr(bf,
                               population_fraction: np.ndarray) -> np.ndarray:
    # This parameters' names come from my original presentation on how to
    # tune consolidation in a system with two binary populations
    # bf-black females (intersectional minority)
    # wf-white females
    # bm-black males
    # wm-white males (intersectional majority)
    wf = population_fraction[1, 0] - bf
    bm = population_fraction[0, 0] - bf
    wm = 1 + bf - population_fraction[1, 0] - population_fraction[0, 0]
    return np.array([[bf, bm], [wf, wm]])


def consol_comp_pop_frac_tnsr(population_fraction: list,
                              consol: int) -> np.ndarray:
    assert 0 <= consol <= 1

    population_fraction = np.array(population_fraction)

    # Verify population fractions are ordered in increasing order
    # Easier to compute next stuff
    for i in population_fraction:
        assert np.all(np.sort(i) == i)

    bf_MC = min(population_fraction[0, 0], population_fraction[1, 0])
    bf_AC = 0.0

    comp_pop_frac_MC = two_bin_comp_pop_frac_tnsr(bf_MC, population_fraction)
    comp_pop_frac_AC = two_bin_comp_pop_frac_tnsr(bf_AC, population_fraction)

    return consol * comp_pop_frac_MC + (1.0-consol) * comp_pop_frac_AC


def is_pop_frac_consistent(pop_fracs_lst, comp_pop_frac_tnsr):

    pop_fracs_lst = np.array(pop_fracs_lst)

    assert np.all(comp_pop_frac_tnsr >= 0)
    assert np.all(comp_pop_frac_tnsr <= 1)

    assert np.all(pop_fracs_lst >= 0)
    assert np.all(pop_fracs_lst <= 1)

    # Check pop_fracs_lst ordered from smaller to larger populations
    # We assume this for several computations, so it is better to be sure
    for pop_fracs in pop_fracs_lst:
        assert np.all(np.sort(pop_fracs) == pop_fracs)

    # Check overall normalization
    for d, pop_frac in enumerate(pop_fracs_lst):
        if np.abs(np.sum(pop_frac) - 1.0) > 1e-10:
            print(f"Bad normalization in simple populations of dim {d}")
            return False

    if np.abs(np.sum(comp_pop_frac_tnsr) - 1.0) > 1e-10:
        print("Bad overall normalization of composite population fractions.")
        return False

    # Check marginals
    assert np.allclose(np.sum(comp_pop_frac_tnsr, axis=1), pop_fracs_lst[0])
    assert np.allclose(np.sum(comp_pop_frac_tnsr, axis=0), pop_fracs_lst[1])

    return True


def am_preliminary_checks(h_mtrx_lst, comp_pop_frac_tnsr, pop_fracs_lst):
    h_mtrx_lst = np.array(h_mtrx_lst)
    # Check range of parameters
    assert np.all(h_mtrx_lst <= 1)
    assert np.all(h_mtrx_lst >= 0)

    if pop_fracs_lst is not None:
        assert is_pop_frac_consistent(pop_fracs_lst, comp_pop_frac_tnsr)
    else:
        assert np.abs(np.sum(comp_pop_frac_tnsr) - 1.0) < 1e-13


def make_composite_index(g_vec):
    assert all(g_vec) > 0
    elems = [list(range(g)) for g in g_vec]
    comp_indices = itertools.product(*elems)
    return list(comp_indices)


def build_probs_pop(comp_pop_frac_tnsr):
    g_vec = comp_pop_frac_tnsr.shape
    X = make_composite_index(g_vec)
    p = []
    for i_vec in X:
        p.append(comp_pop_frac_tnsr[i_vec])
    assert np.abs(np.sum(p) - 1.0) < 1e-13
    return X, p


def build_social_structure(N, comp_pop_frac_tnsr, directed=None):
    # Build probability distribution of composite populations
    memberships, probs = build_probs_pop(comp_pop_frac_tnsr)
    # Assign a membership to each node
    G = nx.Graph()
    for n in range(N):
        node_type = memberships[np.random.choice(len(memberships), p=probs)]
        G.add_node(n, attr=node_type)
    return G


def am_v2(
        homophily,
        consolidation_param,
        directed=False,
        marginal_distribution=None,
        m=3,
        N=1000,
        v=0,
        **kwargs):

    # Centola style connections

    comp_pop_frac_tnsr = consol_comp_pop_frac_tnsr(
        marginal_distribution,
        consolidation_param)

    h1 = np.array([[homophily, 1-homophily], [1-homophily, homophily]])
    h2 = h1.copy()
    h_mtrx_lst = np.array([h1, h2])

    h_mtrx_lst = np.array(h_mtrx_lst)

    # Assert that every parameter is within appropriate ranges
    am_preliminary_checks(
        h_mtrx_lst,
        comp_pop_frac_tnsr,
        pop_fracs_lst=marginal_distribution)

    # Compute number of dimensions
    D = len(h_mtrx_lst)
    assert D == comp_pop_frac_tnsr.ndim

    # Build random population of nodes alla Centola
    G = build_social_structure(N, comp_pop_frac_tnsr, directed)

    # Build interaction function (faster than an if-switch inside
    # the inner for loop)
    interaction = interaction_all
    normalization = interaction_all_normalization(h_mtrx_lst)

    # Iterate
    h_vec = np.zeros(D)
    n_lnks = 0

    while n_lnks < N*m:
        # for i in range(N*m):
        if v == 1 and n_lnks % 1000 == 0:
            print(n_lnks)
        # Random node 1 and 2
        n, target = np.random.randint(N, size=2)
        if n == target:
            continue
        # Check if link exists
        if G.has_edge(n, target):
            continue
        # Compute homophily
        orig_idx = G.nodes[n]["attr"]
        target_idx = G.nodes[target]["attr"]

        for d in range(D):
            h_vec[d] = h_mtrx_lst[
                    d,
                    orig_idx[d],
                    target_idx[d]
                    ]

        # Check if the tie is made
        successful_tie = interaction(h_vec, normalization)

        # Create links
        if successful_tie:
            G.add_edge(n, target)
            n_lnks += 1

    return G


def social_origins_network(
        N=3200,
        m=5,
        alpha=1.0,
        beta=1.0,
        H=32,
        D=10,
        dimension_aggrigation=min,
        **kwargs):

    G = nx.Graph()

    max_dist = math.ceil(np.log2(H)) + 1
    beta_vec = np.array([np.exp(-beta * x) for x in range(1, max_dist + 1)])
    beta_vec = beta_vec / np.sum(beta_vec)

    alpha_vec = np.array([np.exp(-alpha * x) for x in range(1, max_dist + 1)])
    alpha_vec = alpha_vec / np.sum(alpha_vec)

    def attributes_dist(a1, a2):
        for i in range(max_dist - 1, -1, -1):
            if a1//(2**i) != a2//(2**i):
                return i + 2
        return 1

    attribute_dist_matrix = np.zeros((H, H), dtype=np.int16)
    for i in range(H):
        for j in range(H):
            attribute_dist_matrix[i, j] = attributes_dist(i, j)

    # CREATE NODES
    for i in range(N):
        attributes = []
        d1 = np.random.randint(H)
        attributes.append(d1)

        dist = np.random.choice(range(1, max_dist + 1), D - 1,
                                p=beta_vec, replace=True)

        for j in range(0, D - 1):
            possible_attributes = [x for x in range(H)
                                   if attribute_dist_matrix[x, d1] == dist[j]]

            attributes.append(np.random.choice(possible_attributes))

        G.add_node(i, attr=tuple(attributes))

    # CREATE NODES DISTANCE MATRIX
    node_dist_matrix = np.zeros((N, N), dtype=np.int16)
    for i in range(N):
        for j in range(N):
            attr_i = G.nodes[i]["attr"]
            attr_j = G.nodes[j]["attr"]
            distances = [attribute_dist_matrix[attr_i[k], attr_j[k]]
                         for k in range(D)]
            node_dist_matrix[i, j] = dimension_aggrigation(distances)

    # CREATE LINKS
    n_links = 0
    it_last_update = 0
    while n_links < N * m:
        i = np.random.randint(N)
        dist = np.random.choice(range(1, max_dist + 1), p=alpha_vec)

        candidate_nodes = np.where(node_dist_matrix[i, :] == dist)[0]

        if len(candidate_nodes) == 0:
            continue

        selected_node = np.random.choice(candidate_nodes)
        if not G.has_edge(i, selected_node):
            G.add_edge(i, selected_node)
            n_links += 1
            it_last_update = 0
        else:
            it_last_update += 1

            if it_last_update > 100:    # 100 is arbitrary
                return None

    return G


generator_name_mapping = {
    "am_v2": am_v2,
    "social_origins_network": social_origins_network,
}


def get_network_generator(name):
    if name is None:
        return am_v2
    return generator_name_mapping[name]


def G_attr_to_str(G: nx.Graph, attr):
    G_out = G.copy()
    for n in G_out.nodes():
        G_out.nodes[n][attr] = str(G_out.nodes[n][attr])
    return G_out
