import matplotlib.pyplot as plt
import numpy as np
import network_generation as homomul


def fig_colored_matrix(M,
                       ax=None,
                       xticks=None,
                       yticks=None,
                       show_colorbar=False,
                       figsize=None,
                       vmin=0,
                       vmax=1):

    if ax:
        plt.sca(ax)
    else:
        if not figsize:
            nx = M.shape[0]
            ny = M.shape[1]
            figsize = (nx, ny*3.0/4.0)
        plt.figure(figsize=figsize)
        ax = plt.axes()

    if vmin is None:
        vmin = np.min(M)
    if vmax is None:
        vmax = np.max(M)
    plt.imshow(M, vmin=vmin, vmax=vmax)

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[j, i] >= 0.5*(vmax-vmin):
                plt.text(i, j, f"{M[j,i]:.02f}",
                         color="k",
                         weight="bold",
                         ha="center",
                         va="center")
            else:
                plt.text(i, j, f"{M[j,i]:.02f}",
                         color="w",
                         weight="bold",
                         ha="center",
                         va="center")

    if not xticks:
        xticks = np.arange(M.shape[0])
    if not yticks:
        yticks = np.arange(M.shape[1])

    plt.xticks(range(M.shape[0]), xticks)
    plt.yticks(range(M.shape[1]), yticks)

    if show_colorbar:
        plt.colorbar()

    return ax

##############################################################################
##############################################################################
# 2 binary attributes
##############################################################################
##############################################################################


def fig_2bin_H_comp_and_simple(h_mtrx_lst, H_comp):
    assert len(h_mtrx_lst) == 2
    assert h_mtrx_lst[0].shape == (2, 2)
    assert h_mtrx_lst[1].shape == (2, 2)

    g_vec = [len(h) for h in h_mtrx_lst]
    comp_indices = homomul.make_composite_index(g_vec)

    fig = plt.figure(figsize=(7*3.0/4, 4*3.0/4), constrained_layout=True)
    spec = fig.add_gridspec(2, 2, width_ratios=[1, 2], height_ratios=[1, 1])

    ax_ul = fig.add_subplot(spec[0, 0])
    ax_bl = fig.add_subplot(spec[1, 0])
    ax_r = fig.add_subplot(spec[:, 1])

    ax_l = [ax_ul, ax_bl]

    for i, h_mtrx_i in enumerate(h_mtrx_lst):
        fig_colored_matrix(h_mtrx_i, ax=ax_l[i])

    fig_colored_matrix(H_comp,
                       ax=ax_r,
                       xticks=comp_indices,
                       yticks=comp_indices,
                       show_colorbar=True)

    return fig


def fig_2bin_comp_pop_frac(comp_pop_frac_tnsr):
    assert comp_pop_frac_tnsr.shape == (2, 2)

    fig = plt.figure(figsize=(3, 3), constrained_layout=True)
    spec = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[1, 3])

    ax_u = fig.add_subplot(spec[0, 0])
    ax_r = fig.add_subplot(spec[1, 1])
    ax_c = fig.add_subplot(spec[1, 0], sharex=ax_u, sharey=ax_r)

    plt.sca(ax_u)
    plt.bar([0, 1], np.sum(comp_pop_frac_tnsr, axis=0), color="grey")
    plt.ylim(0, 1)
    plt.setp(ax_u.get_xticklabels(), visible=False)
    for i, yi in enumerate(np.sum(comp_pop_frac_tnsr, axis=0)):
        plt.text(i, 0.25, f"{yi:.02f}",
                 color="k",
                 weight="bold",
                 ha="center",
                 va="center")

    plt.sca(ax_r)
    plt.barh([0, 1], np.sum(comp_pop_frac_tnsr, axis=1), color="grey")
    plt.xlim(0, 1)
    plt.setp(ax_r.get_yticklabels(), visible=False)
    for i, yi in enumerate(np.sum(comp_pop_frac_tnsr, axis=1)):
        plt.text(0.25, i, f"{yi:.02f}",
                 color="k",
                 weight="bold",
                 ha="center",
                 va="center",
                 rotation=270)

    fig_colored_matrix(comp_pop_frac_tnsr,
                       ax=ax_c,
                       show_colorbar=False)

    return fig


def fig_2attr_heatmap(key1, values1, key2, values2, result, title=None):
    plt.imshow(result, interpolation='nearest')
    plt.yticks(np.arange(values1.size), [f'{x:0.1f}' for x in values1])
    plt.xticks(np.arange(values2.size), [f'{x:0.1f}' for x in values2])
    plt.ylabel(key1)
    plt.xlabel(key2)
    plt.colorbar()
    if title:
        plt.title(title)
    plt.show()
