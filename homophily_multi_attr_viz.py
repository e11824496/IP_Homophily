import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import numpy as np
import imp
homomul = imp.load_source("homomul","2021_07_12_homophily_multiple_attributes.py")
# fair_simul = imp.load_source("fair_simul","2022_11_02_fairness_simulations.py")

##############################################################################
##############################################################################
## General
##############################################################################
##############################################################################

def fig_colored_matrix(
	M,
	ax=None,
	xticks=None,
	yticks=None,
	show_colorbar=False,
	figsize=None,
	vmin=0,
	vmax=1
	):

	if ax:
		plt.sca(ax)
	else:
		if not figsize:
			nx = M.shape[0]
			ny = M.shape[1]
			figsize = (nx,ny*3.0/4.0)
		fig = plt.figure(figsize=figsize)
		ax = plt.axes()

	if vmin is None:
		vmin = np.min(M)
	if vmax is None:
		vmax = np.max(M)
	plt.imshow(M,vmin=vmin,vmax=vmax)
	
	for i in range(M.shape[0]):
		for j in range(M.shape[1]):
			if M[j,i] >= 0.5*(vmax-vmin):
				plt.text(i,j,f"{M[j,i]:.02f}",
						color="k",
						weight="bold",
						ha="center",
						va="center")
			else:
				plt.text(i,j,f"{M[j,i]:.02f}",
						color="w",
						weight="bold",
						ha="center",
						va="center")
	
	if not xticks:
		xticks = np.arange(M.shape[0])
	if not yticks:
		yticks = np.arange(M.shape[1])

	plt.xticks(range(M.shape[0]),xticks)
	plt.yticks(range(M.shape[1]),yticks)

	if show_colorbar:
		plt.colorbar()

	return ax

##############################################################################
##############################################################################
## 2 binary attributes
##############################################################################
##############################################################################

def fig_2bin_H_comp_and_simple(h_mtrx_lst,H_comp):
	
	assert len(h_mtrx_lst) == 2
	assert h_mtrx_lst[0].shape == (2,2)
	assert h_mtrx_lst[1].shape == (2,2)

	g_vec = [len(h) for h in h_mtrx_lst]
	comp_indices = homomul.make_composite_index(g_vec)

	fig = plt.figure(figsize=(7*3.0/4,4*3.0/4), constrained_layout=True)
	spec = fig.add_gridspec(2, 2,width_ratios=[1,2],height_ratios=[1,1])

	ax_ul = fig.add_subplot(spec[0, 0])
	ax_bl = fig.add_subplot(spec[1, 0])
	ax_r = fig.add_subplot(spec[:, 1])

	ax_l = [ax_ul,ax_bl]

	for i, h_mtrx_i in enumerate(h_mtrx_lst):
		fig_colored_matrix(h_mtrx_i,ax=ax_l[i])

	fig_colored_matrix(H_comp,
								   ax=ax_r,
								   xticks=comp_indices,
								   yticks=comp_indices,
								   show_colorbar=True)

	return fig

def fig_2bin_comp_pop_frac(comp_pop_frac_tnsr):
	
	assert comp_pop_frac_tnsr.shape == (2,2)

	# gs_kw = dict(width_ratios=[2, 1], height_ratios=[1, 2])
	# fig, axs = plt.subplots(2,2,figsize=(6,6),gridspec_kw=gs_kw,constrained_layout=True)

	fig = plt.figure(figsize=(3,3), constrained_layout=True)
	spec = fig.add_gridspec(2, 2,width_ratios=[3,1],height_ratios=[1,3])

	ax_u = fig.add_subplot(spec[0, 0])
	ax_r = fig.add_subplot(spec[1, 1])
	ax_c = fig.add_subplot(spec[1, 0],sharex=ax_u,sharey=ax_r)

	plt.sca(ax_u)
	plt.bar([0,1],np.sum(comp_pop_frac_tnsr,axis=0),color="grey")
	plt.ylim(0,1)
	# plt.xticks([0,1],[])
	plt.setp( ax_u.get_xticklabels(), visible=False)
	for i,yi in enumerate(np.sum(comp_pop_frac_tnsr,axis=0)):
		plt.text(i,0.25,f"{yi:.02f}",
						color="k",
						weight="bold",
						ha="center",
						va="center")

	plt.sca(ax_r)
	plt.barh([0,1],np.sum(comp_pop_frac_tnsr,axis=1),color="grey")
	plt.xlim(0,1)
	# plt.xticks([0,1],[])
	plt.setp( ax_r.get_yticklabels(), visible=False)
	for i,yi in enumerate(np.sum(comp_pop_frac_tnsr,axis=1)):
		plt.text(0.25,i,f"{yi:.02f}",
						color="k",
						weight="bold",
						ha="center",
						va="center",
						rotation=270)


	fig_colored_matrix(comp_pop_frac_tnsr,
					   ax=ax_c,
					   show_colorbar=False)

	# asp = np.diff(ax_u.get_xlim())[0] / np.diff(ax_r.get_ylim())[0]
	# ax_c.set_aspect(asp)

	return fig

def fig_lst_2bin_comp_pop_frac(
	consol_lst,
	f0_0_lst,
	f1_0_lst,
	save_path
	):
	"""
	Plot the population distribution for a list of correlation (consolidation)
	values and minority fractions of each of the two populations.
	"""
	for consol in consol_lst:
		for f0_0 in f0_0_lst:
			for f1_0 in f1_0_lst:
				pop_fracs_lst = [[f0_0,1-f0_0], [f1_0,1-f1_0]]
				comp_pop_frac_tnsr = homomul.consol_comp_pop_frac_tnsr(pop_fracs_lst,consol)
				fig = fig_2bin_comp_pop_frac(comp_pop_frac_tnsr)
				plt.savefig(save_path+f"f0_0{f0_0:.02f}_f1_0{f1_0:.02f}_k{consol:.02f}.pdf")
				plt.savefig(save_path+f"f0_0{f0_0:.02f}_f1_0{f1_0:.02f}_k{consol:.02f}.png",dpi=300,transparent=True)	
				plt.close()

def fig_lst_2bin_H_comp_and_simple(
	h_mtrx_lst_LST,
	kind,
	save_path,
	**kwargs,
	):
	"""
	Plot the homophily matrices for a list of 1d homophily matrix lists
	and interaction kinds.
	"""
	for i, h_mtrx_lst in enumerate(h_mtrx_lst_LST):
		H_theor = homomul.composite_H(
			h_mtrx_lst,
			kind,
			**kwargs
			)
		fig = fig_2bin_H_comp_and_simple(h_mtrx_lst,H_theor)
		plt.savefig(save_path+f"hmtrx_{i}.pdf")
		plt.savefig(save_path+f"hmtrx_{i}.png",dpi=300,transparent=True)	
		plt.close()

##############################################################################
##############################################################################
## Fairness / inequality
##############################################################################
##############################################################################

##############################################################################
## Theil
##############################################################################

def ax_rel_abs_thiel(
	## Data
	x ,
	theil_total,
	theil_bet,
	theil_groups_wit,
	theil_groups,
	## Plot tweak
	xlabel,
	markers_groups = ["o","s","v","P","X","d","p"],
	lines_groups = ["-","--",":","-."],
	show_l_y = True,
	show_r_y = True,
	show_legend=False,
	group_labels=None,
	## Axes
	ax = None,
	ylim=None,
	):
	"""
	To plot the relaive contribution of "between" and "within" components of 
	Theil index together with the absolute value of total Theil and the Theil
	associated to each group.
	"""
	## Ensure I have enough markers to plot all different lines
	if len(markers_groups) < len(theil_groups):
		raise Exception(f"We need {len(theil_groups)-len(markers_groups)} more markers in the markers_groups list.")
	if len(lines_groups) < len(theil_groups):
		raise Exception(f"We need {len(theil_groups)-len(lines_groups)} more markers in the markers_groups list.")

	## Compute the relative contributions to Theil index
	theil_bet_rel = theil_bet/theil_total
	theil_groups_wit_rel = [i/theil_total for i in theil_groups_wit]

	theil_contribs = [theil_bet_rel] + theil_groups_wit_rel
	for i,theil_contrib_i in enumerate(theil_contribs):
		theil_contribs[i] = np.array(theil_contrib_i)
		theil_contribs[i][np.isnan(theil_contrib_i)] = 0 ## Otherwise the plot shows some white spaces

	## Start plotting
	if ax is None:
		ax0 = plt.axes()
	else:
		ax0 = ax
	plt.sca(ax0)
	## Absolute value of total inequality
	plt.plot(x,theil_total,"-",lw=2,color="k")
	if show_legend:
		plt.plot([],[],"-k",lw=2,label="Total")
	## Absolute value of inequality within each group(?)
	for i,theil_i in enumerate(theil_groups):
		mrkr = markers_groups[i]
		ls = lines_groups[i]
		# plt.plot(x,theil_i,f"--{mrkr}k",ms=2,lw=1)
		plt.plot(x,theil_i,"k",lw=1,ls=ls)
	if show_legend:
		for i, g in enumerate(group_labels):
			ls = lines_groups[i]
			plt.plot([],[],"k",lw=1,ls=ls,label=g)
		plt.legend(
			frameon=False,
			loc='lower right',
			bbox_to_anchor=(1.0, 1.0),
          	ncol=1
			)
	if show_l_y:
		plt.ylabel("Theil")
	else:
		# ax0.yaxis.set_ticklabels([])
		plt.setp(ax0.get_yticklabels(), visible=False)
	if ylim is not None:
		plt.ylim(ylim)
	ax0.set_zorder(20)
	ax0.patch.set_alpha(0.0)
	plt.xlabel(xlabel)

	ax1 = ax0.twinx()
	## Relative contribution to inequality of each group
	colors = ["grey"] + plt.rcParams['axes.prop_cycle'].by_key()['color'] ## Grey + standard cycle
	plt.stackplot(x,theil_contribs,alpha=0.5,colors=colors)
	if show_legend:
		plt.fill_between([],[],[],alpha=0.5,color="grey",label="Between")
		for i,g in enumerate(group_labels):
			plt.fill_between([],[],[],alpha=0.5,label=g,color=colors[i+1])
		plt.legend(
			frameon=False,
			loc='lower left', 
			bbox_to_anchor=(0.0, 1.0),
          	ncol=1
			)
	if show_r_y:
		plt.ylabel("Theil share")
	else:
		# ax1.yaxis.set_ticklabels([])
		plt.setp(ax1.get_yticklabels(), visible=False)
	plt.ylim(0,1)
	ax1.set_zorder(0)
	
	# ax1.yaxis.set_label_position("left")
	# ax1.yaxis.tick_left()

	# ax0.yaxis.set_label_position("right")
	# ax0.yaxis.tick_right()

	return ax0, ax1

def build_all_ax_theil(
	fairness_info_df,
	x_col,
	g_vec,
	xlabel=None,
	show_legends=False,
	ax_lst = None,
	show_l_y=False,
	show_r_y=False,
	ylim=None,
	label_plots=False,
	bottom_zero=False
	):
	"""
	To build all the Theil inequality plots for each dimension and also for the
	multidimensional groups.

	fairness_info_df - Dataframe with Theil and model parameters
	x_col - Column to use as x axis
	g_vec - 1D array with number of groups in each dimension
	"""

	x = fairness_info_df[x_col]

	if show_legends:
		## Dimension-wise groups
		max_g = max(g_vec)
		dim_gr_lbls = list(range(max_g))
		## Multidimensional groups
		multi_gr_lbls = homomul.make_composite_index(g_vec)
		## Scale figure size
		fig_scl = (1+len(multi_gr_lbls))/3.0
	else:
		dim_gr_lbls = None
		multi_gr_lbls = None
		fig_scl = 1.0

	if ax_lst is None:
		plt.figure(figsize=(0.4*8*(len(g_vec)+1),0.4*6*fig_scl))
	
	ax_lst_fnl = []
	for d,v_d in enumerate(g_vec):
		## Get the data for 1D Theil
		theil_total = fairness_info_df[f"dim{d}_theil"]
		theil_bet = fairness_info_df[f"dim{d}_theil_bet"]
		theil_groups_wit = []
		theil_groups = []
		for g in range(v_d):
			theil_groups_wit.append(fairness_info_df[f"dim{d}_g{g}_theil_wit"])
			theil_groups.append(fairness_info_df[f"dim{d}_g{g}_theil"])

		## Draw the axes for dimension-wise Theil index
		if ax_lst is None:
			show_r_y = False
			if d==0:
				ax = plt.subplot(1,len(g_vec)+1,d+1)
				show_legend = show_legends
				xlabel=xlabel
			else:
				ax = plt.subplot(1,len(g_vec)+1,d+1,sharey=ax)
				show_legend = False
				xlabel=None
		else:
			ax = ax_lst[d]
			show_legend = False
			if d>0:
				## For the case where I have already configured a shared y axis		
				try:
					ax.sharey(ax_lst[0])
				except ValueError:
					pass
		show_l_y = show_l_y and d==0
		plt.sca(ax)
		ax_abs, ax_rel = ax_rel_abs_thiel(
			## Data
			x,
			theil_total,
			theil_bet,
			theil_groups_wit,
			theil_groups,
			## Plot tweak
			xlabel=xlabel,
			show_l_y=show_l_y,
			show_r_y=False,
			show_legend=show_legend,
			group_labels=dim_gr_lbls,
			## Axes
			ax=ax,
			ylim=ylim,
			)
		if label_plots:
			plt.title(f"Dim {d}",color="grey")
		ax_lst_fnl.append(ax_abs)

	## Get the data for multidimensional Theil
	theil_total =fairness_info_df["multi_theil"]
	theil_bet = fairness_info_df["multi_theil_bet"]
	
	comp_indices = homomul.make_composite_index(g_vec)
	theil_groups_wit = []
	theil_groups = []
	for I,g in enumerate(comp_indices):
		theil_groups.append(fairness_info_df[f"multi{g}_theil"])
		theil_groups_wit.append(fairness_info_df[f"multi{g}_theil_wit"])

	## Draw the axes for multidimensional Theil
	if ax_lst is None:
		ax = plt.subplot(1,len(g_vec)+1,len(g_vec)+1,sharey=ax)
		show_r_y=True
		show_l_y=False
	else:
		ax = ax_lst[-1]
		## For the case where I have already configured a shared y axis
		try:
			ax.sharey(ax_lst[0])
		except ValueError:
			pass
	plt.sca(ax)
	ax_abs, ax_rel = ax_rel_abs_thiel(
			## Data
			x,
			theil_total,
			theil_bet,
			theil_groups_wit,
			theil_groups,
			## Plot tweak
			xlabel=None,
			show_r_y = show_r_y,
			show_l_y = False,
			show_legend=show_legends,
			group_labels=multi_gr_lbls,
			## Axes
			ax = ax,
			ylim=ylim,
			)
	if label_plots:
		plt.title("Multi",color="grey")
	ax_lst_fnl.append(ax_abs)

	for axi in ax_lst_fnl[1:]:
		plt.setp(axi.get_yticklabels(),visible=False)
		if bottom_zero: ## To make sure the bottom of the plot is always 0
			ylim = axi.get_ylim()
			axi.set_ylim(0,ylim[1]) 

	# if ax_lst is not None:
	# 	y_max = 0
	# 	for axi in ax_lst:
	# 		ylim_i = axi.get_ylim()
	# 		y_max = max(y_max,ylim_i[1])
	# 		y_min = ylim_i[0]
	# 	for axi in ax_lst:
	# 		axi.set_ylim((y_min,y_max))
	# if ax_lst is not None:
	# 	for axi in ax_lst[1:]:
	# 		axi.sharey(ax_lst[0])
			## Hide title and ticks again
			# axi.yaxis.set_ticklabels([])
			# axi.ylabel()

	return ax_lst

##############################################################################
## Delta / A12
##############################################################################

def build_all_ax_delta_1vR(
	## Data
	fairness_info_df,
	x_col,
	g_vec,
	## Plot tweak
	xlabel=None,
	label_lines=False,
	show_percentages=False,
	lines_groups = ["-","--",":","-."],
	label_plots=False,
	## Axes
	ax_lst = None,
	):
	"""
	To build all the delta inequality plots for each dimension and also for the
	multidimensional groups.

	fairness_info_df - Dataframe with Theil and model parameters
	x_col - Column to use as x axis
	g_vec - 1D array with number of groups in each dimension
	"""
	G = 1
	for v_d in g_vec:
		G *= v_d
	if len(lines_groups) < G:
		raise Exception(f"We need {len(theil_groups)-len(lines_groups)} more markers in the markers_groups list.")

	x = fairness_info_df[x_col]
	if ax_lst is None:
		plt.figure(figsize=(0.4*8*len(g_vec),0.4*6))

	for d,v_d in enumerate(g_vec):
		if ax_lst is None:
			ax = plt.subplot(1,len(g_vec)+1,d+1)
			if d == 0:
				plt.ylabel("$\delta_{iu}$")
				plt.xlabel(xlabel)
			else:
				ax.yaxis.set_ticklabels([])
		else:
			ax = ax_lst[d]
			plt.sca(ax)
		ax.spines.right.set_visible(False)
		ax.spines.top.set_visible(False)
		for i in range(v_d):
			y = fairness_info_df[f"dim{d}_g{i}_delta"]
			ls = lines_groups[i]
			p = plt.plot(x,y,ls=ls)
			plt.ylim(-1.05,1.05)
			## Label each line
			## Initialize label with empty string and update with info
			lbl_str = ""
			if label_lines:
				lbl_str += f"{i}"
			if show_percentages:
				pass ## TO DO: Include this for dimensionwise populations?
			c = p[0].get_color()
			ax.annotate(
					lbl_str,
					(
					#1.01,
					1.01*x.iloc[-1], 
					y.iloc[-1]),
                    # xycoords=('axes fraction', 'data'), 
                    xycoords=('data', 'data'), 
                    color=c,
	                va="center",
	                ha="left")
		if label_plots:
			plt.title(f"Dim {d}",color="grey")

	if ax_lst is None:
		ax = plt.subplot(1,len(g_vec)+1,len(g_vec)+1)
		ax.yaxis.set_ticklabels([])
	else:
		ax = ax_lst[-1]
		plt.sca(ax)
	ax.spines.right.set_visible(False)
	ax.spines.top.set_visible(False)

	## Preliminary verifications and data extraction to show percentages in lines
	if show_percentages:
		pop_cols = [col for col in fairness_info_df.columns.values if col.startswith("pop")]
		if len(pop_cols) == 0:
			print ("WARNING! No column in dataframe starts with name 'pop'. Won't show show percentages.")
		else:
			## Make sure we have the exact same proportions in each raw,
			## otherwise this doesn't make sense
			for col in pop_cols:
				unique_vals = fairness_info_df[col].unique()
				assert len(unique_vals) == 1
			## Since all the rows are the same, get the first
			pop_prop_df = fairness_info_df[pop_cols].iloc[0]

	comp_indices = homomul.make_composite_index(g_vec)
	for I, g in enumerate(comp_indices):
		y = fairness_info_df[f"multi{g}_delta"]
		ls = lines_groups[I]
		p = plt.plot(x,y,ls=ls)
		plt.ylim(-1.05,1.05)
		lbl_str = ""
		if label_lines:
			lbl_str += f"{g}"
		if show_percentages and len(pop_cols)>0:
			pop_prop_I = 100*pop_prop_df[f"pop{g}"]
			lbl_str += str(int(pop_prop_I))

		c = p[0].get_color()
		ax.annotate(
			lbl_str,
			(
			#1.01,
			1.01*x[~np.isnan(y)].iloc[-1], 
			y[~np.isnan(y)].iloc[-1]),
            # xycoords=('axes fraction', 'data'), 
            xycoords=('data', 'data'), 
            color=c,
            va="center",
            ha="left")
	if label_plots:
		plt.title("Multi",color="grey")
	return ax_lst

##############################################################################
## Multi-panel figure
##############################################################################

def fig_ineq_multiple_dependence(
	## Data
	df,
	g_vec,
	x_axis,
	x_grid,
	y_grid,
	inequality,
	group_cols=None,
	## Plot tweak
	xlabel=None,
	ylim=None,
	y_scale="relative",
	show_zoom_lines=True,
	label_plots=True,
	y_grid_label="",
	x_grid_label="",
	theil_bottom_zero=False,
	**kwargs
	):
	"""
	Visualize the dependence of inequality on multiple model parameters.
	"""

	## Verify that the only parameters with several values are x_axis, x_grid,
	## and y_grid. Otherwise we would be aggregating results from simulations
	## with different parameters instead of repetitions of the same simulation.
	## Also store the unique values of each parameter
	param_vals = {}
	for col in df:
		if col.startswith("prm_"):
			unique_vals = df[col].unique()
			unique_vals.sort()
			num_unique_vals = len(unique_vals)
			if num_unique_vals > 1 and col[4:] not in [x_axis,x_grid,y_grid]:
				raise Exception(f"Parameter {col} wasn't taken into account and takes different values.")
			param_vals[col] = unique_vals

	## Aggregate the results
	if group_cols is None:
		group_cols = [col for col in df.columns.values if col.startswith("prm_")]
	aggr_res_df = fair_simul.aggr_repeated_simul(
		df,
		aggr_fun_lst=[np.mean],
		group_cols=group_cols)

	## Generate the structure of the figure
	if y_scale == "relative":
		sharexy = "none"
	elif y_scale == "absolute":
		sharexy = "all"	
	else:
		raise ValueError(f"Unsupported {y_scale} y_scale.")
	height_ratios = [10,0.5]+[10]*len(param_vals["prm_"+y_grid])
	n_pnls = len(g_vec)+1 ## Number of panels per simulation
	fig, axs = plt.subplots(
		sharex=sharexy,
		sharey=sharexy,
		ncols=n_pnls*len(param_vals["prm_"+x_grid]), 
		nrows=len(param_vals["prm_"+y_grid])+2, ## I sum 1 for the legend and 1 for the suptitles
		figsize=(.27*8*n_pnls*len(param_vals["prm_"+x_grid]), .27*6*(len(param_vals["prm_"+y_grid])+1)),
        layout="constrained",
        squeeze=True,
        gridspec_kw = {'height_ratios':height_ratios})
	if y_scale == "relative":
		plt.subplots_adjust(wspace=0.5)
	n_sbplts = len(g_vec) + 1 ## Number of panels per system
	for xi, x_prm in enumerate(param_vals["prm_"+x_grid]):
		msk_x = aggr_res_df[x_grid] == x_prm
		for yi, y_prm in enumerate(param_vals["prm_"+y_grid]):
			## Extract the data to plot
			msk_y = aggr_res_df[y_grid] == y_prm
			msk = np.logical_and(msk_x,msk_y)
			res_plot_df = aggr_res_df.loc[msk]
			## Rename results columns to remove "mean" suffix
			res_plot_df.columns = [col[:-5] if col.endswith("_mean") else col for col in res_plot_df.columns.values]
			## Setup the axes list to plot the results into
			# ax_lst = [axs[yi+1,xi*3],axs[yi+1,xi*3+1],axs[yi+1,xi*3+2]]
			ax_lst = [axs[yi+2,xi*n_sbplts+sbplt_i] for sbplt_i in range(n_sbplts)]
			## Hide x axis
			if yi < len(param_vals["prm_"+y_grid])-1:
				for ax in ax_lst:
					plt.setp( ax.get_xticklabels(), visible=False)
			## Show axis conditionally
			if xi == 0:
				show_l_y=True
				show_r_y=False
			elif xi == len(param_vals["prm_"+x_grid])-1:
				show_l_y=False
				show_r_y=True
			else:
				show_l_y=False
				show_r_y=False
			## Plot
			if inequality == "Theil":
				build_all_ax_theil(
					res_plot_df,
					x_col=x_axis,
					g_vec=g_vec,
					xlabel=None,
					show_legends=False,
					ax_lst = ax_lst,
					show_l_y = show_l_y,
					show_r_y = show_r_y,
					ylim=ylim,
					bottom_zero=theil_bottom_zero
					)
			elif inequality == "delta":
				build_all_ax_delta_1vR(
					## Data
					res_plot_df,
					x_col=x_axis,
					g_vec=g_vec,
					## Plot tweak
					xlabel=None,
					label_lines=False,
					lines_groups = ["-","--",":","-."],
					## Axes
					ax_lst = ax_lst,
					**kwargs
					)
			else:
				raise ValueError(f"Inequality metric {inequality} not implemented.")
			if label_plots:
				if yi == 0:
					for i,ax in enumerate(ax_lst[:-1]):
						ax.set_title(f"Dim {i}",color="grey")
					ax_lst[-1].set_title("Multi",color="grey")
	
	## For plot suptitles
	gs = axs[0,0].get_gridspec()
	if label_plots:
		ineq_label = {
			"Theil":"Theil",
			"delta":"$\delta_{iu}$"
			}
		for xi, x_prm in enumerate(param_vals["prm_"+x_grid]):
			ax_lst = [axs[1,xi*n_sbplts+sbplt_i] for sbplt_i in range(n_sbplts)]
			## Remove the underlying axes
			for ax in ax_lst:
			    ax.remove()
			## Create new axes
			axbig = fig.add_subplot(gs[1, xi*n_sbplts:(xi*n_sbplts+n_sbplts)])
			axbig.axis("off")
			axbig.set_title(x_grid_label+"="+str(x_prm))
		for yi, y_prm in enumerate(param_vals["prm_"+y_grid]):
			axs[yi+2,0].set_ylabel(y_grid_label+"="+str(y_prm)+"\n"+ineq_label[inequality])

	## Show relative scales between subplots
	if show_zoom_lines and y_scale=="relative":
		hlpr_show_zoomed_lines(fig,axs)

	## Legend
	## As per https://matplotlib.org/stable/gallery/subplots_axes_and_figures/gridspec_and_subplots.html
	gs = axs[0,0].get_gridspec()
	## Remove the underlying axes
	for ax in axs[0, :]:
	    ax.remove()
	## Create new axes
	axbig = fig.add_subplot(gs[0, :])
	## Remove axis and background
	axbig.axis("off")
	plt.sca(axbig)
	hlpr_make_legend(g_vec,axbig,inequality)
	## Axis labels
	plt.sca(axs[-1,0])
	plt.xlabel(xlabel)
	if inequality == "delta" and not label_plots:
		plt.ylabel("$\delta_{iu}$")
	return fig

def hlpr_show_zoomed_lines(fig,axs):
	## Fig 7 of https://towardsdatascience.com/5-powerful-tricks-to-visualize-your-data-with-matplotlib-16bc33747e05
	for yi in range(2,axs.shape[0]):
		for xi in range(axs.shape[1]-1):
			maxy1 = axs[yi,xi].get_ylim()[1]
			maxy2 = axs[yi,xi+1].get_ylim()[1]

			if maxy1 != maxy2:
				maxx1 = axs[yi,xi].get_xlim()[1]
				minx2 = axs[yi,xi+1].get_xlim()[0]
				miny1 = axs[yi,xi].get_ylim()[0]
				miny2 = axs[yi,xi+1].get_ylim()[0]
				conn1 = ConnectionPatch(xyA=(maxx1, maxy2), coordsA=axs[yi,xi].transData, 
				                       xyB=(minx2, maxy2), coordsB=axs[yi,xi+1].transData, 
				                       color = 'grey',
				                       alpha=0.5,
				                       zorder=0)
				conn2 = ConnectionPatch(xyA=(maxx1, miny1), coordsA=axs[yi,xi].transData, 
				                       xyB=(minx2, miny2), coordsB=axs[yi,xi+1].transData, 
				                       color = 'grey',
				                       alpha=0.5,
				                       zorder=0)
				fig.add_artist(conn1)
				fig.add_artist(conn2)

def hlpr_make_legend(
	g_vec,
	ax,
	inequality,
	lines_groups = ["-","--",":","-."],
	):
	plt.sca(ax)
	if inequality == "Theil":
		## Configure number of columns
		max_v_d = max(g_vec)
		comp_ind = homomul.make_composite_index(g_vec)
		tot_elems = max_v_d*2+len(comp_ind)*2+2
		n_col = np.ceil((1+max_v_d)/4) + np.ceil(len(comp_ind)/4) ## 6 elements per column (empirical)
		n_col *= 2
		n_empty_dim = 4-(1+max_v_d)%4
		n_empty_multi = 4-len(comp_ind)%4
		if n_empty_dim == 4:
			n_empty_dim = 0
		if n_empty_multi == 4:
			n_empty_multi = 0

		## Lines
		## For every plot
		plt.plot([],[],"k-",lw=3,label="Total")
		## Legend for the dimensions
		for i in range(max_v_d):
			ls = lines_groups[i]
			plt.plot([],[],"k",ls=ls,lw=1,label=i)
		## Add empty legend elements
		for i in range(n_empty_dim):
			plt.plot([],[],color="k",alpha=0.0,label=" ")
		## Legend for the multidimensional groups
		comp_ind = homomul.make_composite_index(g_vec)
		for I, gi in enumerate(comp_ind):
			ls = lines_groups[I]
			plt.plot([],[],"k",ls=ls,lw=1,label=gi)
		## Add empty legend elements
		for i in range(n_empty_multi):
			plt.plot([],[],color="k",alpha=0.0,label=" ")

		## Colors
		## For every plot
		plt.fill_between([],[],[],color="grey",alpha=0.5,label="Between",ec="none")
		## Legend for the dimensions
		for i in range(max_v_d):
			plt.fill_between([],[],[],alpha=0.5,label=i,ec="none")
		## Add empty legend elements
		for i in range(n_empty_dim):
			plt.plot([],[],color="k",alpha=0.0,label=" ")
		## Legend for the multidimensional groups
		## Restart color cycle
		plt.gca().set_prop_cycle(None)
		for I, gi in enumerate(comp_ind):
			plt.fill_between([],[],[],alpha=0.5,label=gi,ec="none")
		## Add empty legend elements
		for i in range(n_empty_multi):
			plt.plot([],[],color="k",alpha=0.0,label=" ")
	elif inequality == "delta":
		## Configure number of columns
		max_v_d = max(g_vec)
		comp_ind = homomul.make_composite_index(g_vec)
		tot_elems = max_v_d*2+len(comp_ind)*2+2
		n_col = np.ceil(max_v_d/4) + np.ceil(len(comp_ind)/4) ## 6 elements per column (empirical)
		n_empty_dim = 4-max_v_d%4
		n_empty_multi = 4-len(comp_ind)%4
		if n_empty_dim == 4:
			n_empty_dim = 0
		if n_empty_multi == 4:
			n_empty_multi = 0

		## Lines
		## Legend for the dimensions
		for i in range(max_v_d):
			ls = lines_groups[i]
			plt.plot([],[],ls=ls,label=i)
		## Add empty legend elements
		for i in range(n_empty_dim):
			plt.plot([],[],alpha=0.0,label=" ")
		## Legend for the multidimensional groups
		plt.gca().set_prop_cycle(None)
		comp_ind = homomul.make_composite_index(g_vec)
		for I, gi in enumerate(comp_ind):
			ls = lines_groups[I]
			plt.plot([],[],ls=ls,label=gi)
		## Add empty legend elements
		for i in range(n_empty_multi):
			plt.plot([],[],alpha=0.0,label=" ")		

	plt.legend(loc="lower left",ncol=int(n_col),frameon=False)