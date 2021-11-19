import numpy as np
from pathlib import Path

from . import utils
from .plotting import plot_links, find_linked_variables, plot_links_metrics
from .variable import Variable_Lev_Metadata
from .constants import ANCIL_FILE

from collections import defaultdict


# Collect results

levels, latitudes, longitudes = utils.read_ancilaries(Path(ANCIL_FILE))
# lat_wts = utils.get_weights(latitudes,norm=True)

def get_parents_from_links(links):
    """ Return a list of variables that are parents, given a
    dictionary of links of the form {variable : [(parent, lag)]}
    """

    linked_variables = set()  # Avoids duplicates and sorts when iterated
    for parents_list in links.values():
        if len(parents_list) > 0:
            for parent in parents_list:
                linked_variables.add(parent[0])  # Remove the lag
    return [(i in linked_variables) for i in range(len(links))]


def record_error(file, error_type, errors):
    """Record if a file has failed with a given error type"""

    error_list = errors.get(error_type, list())
    if file not in error_list:
        error_list.append(file)
        errors[error_type] = error_list


def collect_in_dict(_dict, key, value):
    """Add a value to a list in a dictionary"""

    collected_values = _dict.get(key, list())
    collected_values.append(value)
    _dict[key] = collected_values


def collect_results_file(key, results_file, collected_results, errors, lat_wtg):
    """Collect pcmci results from a file in a dictionary with other
    results.
    
    Parameters:
    -----------
    
    key
        Identifier of the variable and level to which the results will
        be collected.
    results_file
        Path to a file storing results from a pcmci analysis.
    collected_results
        Dictionary. It is updated with the results read when this
        function is called.
    errors
        Auxiliary dictionary to keep track of errors found during the
        execution. It is updated whenever an error is found in a call.
    """

    collected_pc_alpha = collected_results.get(key, dict())
    if not results_file.is_file():
        record_error(results_file, "not_found", errors)
        return
    results = utils.load_results(results_file)
    for pc_alpha, alpha_result in results.items():
        if len(alpha_result) > 0:
            collected = collected_pc_alpha.get(pc_alpha, dict())
            collected_pc_alpha[pc_alpha] = collected
            # Collect parents
            links = alpha_result["links"]
            parents = get_parents_from_links(links)
            parents = [[lat_wtg,0.][parent == False] for parent in parents]
            collect_in_dict(collected, "parents", parents)
            # Collect val_matrix
            val_matrix = alpha_result["val_matrix"]
            collect_in_dict(collected, "val_matrix", val_matrix)
            # Check var_names
            var_names = alpha_result["var_names"]
            if "var_names" not in collected:
                collected["var_names"] = var_names
            else:
                stored_var_names = collected["var_names"]
                assert stored_var_names == var_names, (
                    "Found different variable names.\n"
                    f"Expected {stored_var_names}\nFound {var_names}"
                )
        else:
            record_error(results_file, "is_empty", errors)
    collected_results[key] = collected_pc_alpha
    # Doesn't return, instead modifies the received `collected results` object


def count_total_variables(var_children, levels):
    """ Return the total number of variables given a list of spcam
    variables and a list of levels of for 3D variables
    """
    total = 0
    for child in var_children:

        if child.dimensions == 2:
            total += 1
        elif child.dimensions == 3:
            total += len(levels)
    return total


def collect_results(setup):
    """Collect the pcmci results defined in the setup in a file"""
    collected_results = dict()
    errors = dict()

    total_vars = count_total_variables(setup.spcam_outputs, setup.children_idx_levs)
    total_files = total_vars * len(setup.gridpoints)
    file_progress = 0
    step_progress = 0
    step = 5

    for child in setup.spcam_outputs:
        print(f"Variable: {child.name}")
        if child.dimensions == 2:
            child_levels = [[setup.levels[-1], 0]]
            child_var = Variable_Lev_Metadata(child, None, None)
        elif child.dimensions == 3:
            child_levels = setup.children_idx_levs
        for level in child_levels:
            if child.dimensions == 3:
                child_var = Variable_Lev_Metadata(child, level[0], level[1])
            if setup.analysis == "single":
                for i_grid, (lat, lon) in enumerate(setup.gridpoints):
                    results_file = utils.generate_results_filename_single(
                        child,
                        level[1],
                        lat,
                        lon,
                        setup.ind_test_name,
                        setup.experiment,
                        setup.output_file_pattern,
                        setup.output_folder,
                    )
                    
                    # Area-weights?
                    if setup.area_weighted:
                        lat_wtg = utils.get_weights(setup.region, lat, norm=True)
                    else:
                        lat_wtg = 1.
                        
                    collect_results_file(
                        child_var, results_file, collected_results, errors, lat_wtg
                    )
                    
                    file_progress += 1
                    if file_progress == total_files or (
                        (file_progress / total_files * 100) >= step * step_progress
                    ):
                        step_progress = 1 + np.floor(
                            (file_progress / total_files * 100) / step
                        )
                        print(
                            "Progress: {:.2f}% - {} of {} files".format(
                                file_progress / total_files * 100,
                                file_progress,
                                total_files,
                            )
                        )
            elif setup.analysis == "concat":
                results_file = utils.generate_results_filename_concat(
                    child,
                    level[-1],
                    setup.gridpoints,
                    setup.ind_test_name,
                    setup.experiment,
                    setup.output_file_pattern,
                    setup.output_folder,
                )
                collect_results_file(key, results_file, collected_results, errors)

    return collected_results, errors


def print_errors(errors):
    """Prints the error dictionary according to type"""

    if len(errors) == 0:
        print("No errors were found")
        return

    print("ERRORS\n======")
    for error_type, error_list in errors.items():
        # msg = "{}: {} of {} files ({:.2f}%)".format(
        msg = "{}: {} files".format(
            error_type,
            len(error_list),
            # total_files,
            # len(error_list) / total_files * 100,
        )
        print(msg)
        print("-" * len(msg))
        for file in error_list:
            print(file)


def aggregate_results(collected_results, setup):
    # Configuration
    thresholds = setup.thresholds
    pc_alphas_filter = [str(a) for a in setup.pc_alphas]

    # Combine results
    aggregated_results = dict()
    var_names_parents = None
    for child, collected_pc_alpha in collected_results.items():
        dict_pc_alpha_parents = dict()
        for pc_alpha, collected in collected_pc_alpha.items():
            dict_threshold_parents = dict()
            dict_num_parents = dict()
            dict_per_parents = dict()
            if pc_alpha not in pc_alphas_filter:
                continue  # Skip this pc alpha
            val_matrix_list = np.array(collected["val_matrix"])
            val_matrix_mean = val_matrix_list.mean(axis=0)

            var_names = collected["var_names"]
            if var_names_parents is None:
                # NOTE: This assumes that the list has only one child, and
                # that the child is last
                var_names_parents = var_names[:-1]
            else:
                assert var_names_parents == var_names[:-1], (
                    "Found different variable names. "
                    f"Expected {var_names_parents}\nFound {var_names[:-1]}"
                )

            parents_matrix = collected["parents"]
            parents_matrix = np.array(parents_matrix)
            parents_percent = parents_matrix.mean(axis=0)
            for threshold in thresholds:
                parents_filtered = parents_percent >= threshold
                parents = [
                    i for i in range(len(parents_filtered)) if parents_filtered[i]
                ]
                dict_threshold_parents[str(threshold)] = parents
                dict_num_parents[str(threshold)]  = len(parents)
                dict_per_parents[str(threshold)]  = len(parents) * 100. / len(var_names_parents)
            dict_pc_alpha_parents[pc_alpha] = {
                "parents": dict_threshold_parents,
                "num_parents": dict_num_parents,
                "per_parents": dict_per_parents,
                "val_matrix": val_matrix_mean,
                "parents_percent": parents_percent,
                "var_names": var_names,
            }

        aggregated_results[child] = dict_pc_alpha_parents
    return aggregated_results, var_names_parents


def print_aggregated_results(var_names_parents, aggregated_results):
    """Print combined results"""

    var_names_np = np.array(var_names_parents)
    for child, dict_pc_alpha_parents in aggregated_results.items():
        print(f"\n{child}")
        for pc_alpha, pc_alpha_results in dict_pc_alpha_parents.items():
            print(f"pc_alpha = {pc_alpha}")
            dict_threshold_parents = pc_alpha_results["parents"]
            dict_num_parents = pc_alpha_results["num_parents"]
            dict_per_parents = pc_alpha_results["per_parents"]
            for threshold, parents in dict_threshold_parents.items():
                print(f"* Threshold {threshold}:\n \
                Total number of parents: {dict_num_parents[threshold]} ({dict_per_parents[threshold]:1.1f} %)\n \
                Parents: {var_names_np[parents]}\n\n")


def build_pc_alpha_plot_matrices(var_names_parents, aggregated_results):
    len_parents = len(var_names_parents)
    len_total = len_parents + len(aggregated_results)

    dict_full = dict()
    for i_child, (child, dict_pc_alpha_parents) in enumerate(
        aggregated_results.items()
    ):
        i_child = i_child + len_parents
        for pc_alpha, pc_alpha_results in dict_pc_alpha_parents.items():
            dict_pc_alpha_full = dict_full.get(pc_alpha, dict())
            dict_full[pc_alpha] = dict_pc_alpha_full

            # Build val_matrix (only from inputs to outputs)
            pc_alpha_val_matrix = pc_alpha_results["val_matrix"]
            full_val_matrix = dict_pc_alpha_full.get(
                "val_matrix", np.zeros((len_total, len_total, 1))
            )
            full_val_matrix[:len_parents, i_child, 0] = pc_alpha_val_matrix[
                :len_parents, -1, 1
            ]
            dict_pc_alpha_full["val_matrix"] = full_val_matrix

            # Build link_width
            parents_percent = pc_alpha_results["parents_percent"]
            link_width = dict_pc_alpha_full.get(
                "link_width", np.zeros((len_total, len_total, 1))
            )
            link_width[:len_parents, i_child, 0] = parents_percent[:len_parents]
            dict_pc_alpha_full["link_width"] = link_width

    return dict_full


class Combination:
    """ """

    def __init__(self, pc_alpha, threshold):
        self.pc_alpha = pc_alpha
        self.threshold = threshold

    def __str__(self):
        return f"a{self.pc_alpha}-t{self.threshold}"

    def __repr__(self):
        return repr(str(self))

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return str(self) == str(other)


def build_plot_matrices(var_names_parents, aggregated_results):

    var_names_children = list()

    dict_combinations = dict()
    len_parents = len(var_names_parents)
    len_total = len_parents + len(aggregated_results)
    for i_child, (child, dict_pc_alpha_parents) in enumerate(
        aggregated_results.items()
    ):
        i_child = i_child + len_parents
        var_names_children.append(child)
        for pc_alpha, pc_alpha_results in dict_pc_alpha_parents.items():
            dict_threshold_parents = pc_alpha_results["parents"]
            for threshold, parents in dict_threshold_parents.items():
                key = Combination(pc_alpha, threshold)
                combination_results = dict_combinations.get(key, dict())

                # Build links
                links = combination_results.get(
                    "links", {i: [] for i in range(len_total)}
                )
                links[i_child] = [(parent, 0) for parent in parents]
                combination_results["links"] = links

                # Store results
                dict_combinations[key] = combination_results

    all_var_names = var_names_parents + var_names_children  # Concatenation
    pc_alpha_plot_matrices = build_pc_alpha_plot_matrices(
        var_names_parents, aggregated_results
    )
    for combination, combination_results in dict_combinations.items():
        combination_results["var_names"] = all_var_names
        val_matrix = pc_alpha_plot_matrices[combination.pc_alpha]["val_matrix"]
        combination_results["val_matrix"] = val_matrix
        link_width = pc_alpha_plot_matrices[combination.pc_alpha]["link_width"]
        combination_results["link_width"] = link_width

    return dict_combinations



def build_links_metrics(setup, aggregated_results):

    pc_alphas  = setup.pc_alphas
    outputs_nm = [var.name for var in setup.list_spcam if var.type == "out"]
    
#     dict_combinations = defaultdict(dict)
    dict_combinations = {}
#     dict_combinations = dict.fromkeys(outputs_nm, {})
#     for iVar in dict_combinations: dict_combinations[iVar] = dict.fromkeys(pc_alphas, {})
    
    # Num. of parents (by level in 3D)
    for i_child, (child, dict_pc_alpha_parents) in enumerate(
            aggregated_results.items()
        ):  
        
        for pc_alpha, pc_alpha_results in dict_pc_alpha_parents.items():
            
            child_main = child.var.name
            child_lev  = str(child.level)
            
            if child_main not in dict_combinations: dict_combinations[child_main] = {}
            if pc_alpha not in dict_combinations[child_main]: dict_combinations[child_main][pc_alpha] = {}

            dict_threshold_parents = pc_alpha_results["parents"]
            dict_num_parents       = pc_alpha_results["num_parents"]
            dict_per_parents       = pc_alpha_results["per_parents"]

            thrs = [float(thr) for thr, nPar in dict_num_parents.items()]
            nPar = [nPar       for thr, nPar in dict_num_parents.items()]
            pPar = [pPar       for thr, pPar in dict_per_parents.items()]

            to_be_updated = {child_lev:{'thresholds':thrs,'num_parents':nPar,'per_parents':pPar,}}
            dict_combinations[child_main][pc_alpha].update(to_be_updated)
#             dict_combinations[child_main][float(pc_alpha)][child_lev] = {
#                 'thresholds':thrs,
#                 'num_parents':nPar,
#                 'per_parents':pPar,
#             } 
    
#     print(dict_combinations['tphystnd']['0.001']['992.56']['num_parents'])
#     print(dict_combinations['tphystnd']['0.1']['992.56']['num_parents'])
    
    for child_main in dict_combinations.keys():
        for pc_alpha in dict_combinations[child_main].keys():
            
            if len(dict_combinations[child_main][pc_alpha]) > 1:
            
                nLevs = len(dict_combinations[child_main][pc_alpha].keys())
                nPar_mean = np.zeros([nLevs,len(thrs)])
                pPar_mean = np.zeros([nLevs,len(thrs)])
                count = 0
                for i_child in dict_combinations[child_main][pc_alpha].keys():
                    nPar_mean[count,:] = dict_combinations[child_main][pc_alpha][i_child]['num_parents']
                    pPar_mean[count,:] = dict_combinations[child_main][pc_alpha][i_child]['per_parents'] 
                    count += 1
                nPar_mean, pPar_mean = np.mean(nPar_mean,axis=0), np.mean(pPar_mean,axis=0)
                to_be_updated = {'mean':{'thresholds':thrs,'num_parents':nPar_mean,'per_parents':pPar_mean,}}
                dict_combinations[child_main][pc_alpha].update(to_be_updated)
#                 dict_combinations[child_main][pc_alpha]['mean'] = {
#                     'thresholds':thrs,
#                     'num_parents':nPar_mean,
#                     'per_parents':pPar_mean,
#                 }
            
            else:
                dict_combinations[child_main][str(pc_alpha)]['mean'] = \
                dict_combinations[child_main][str(pc_alpha)].pop('None')
#     print()
#     print(dict_combinations['tphystnd']['0.001']['992.56']['num_parents'])
#     print(dict_combinations['tphystnd']['0.1']['992.56']['num_parents'])
#     print()
#     print(dict_combinations['tphystnd']['0.001']['mean']['num_parents'])
#     print(dict_combinations['tphystnd']['0.1']['mean']['num_parents'])
                
    dict_combinations['mean'] = {}
    for j, iPC in enumerate(pc_alphas):
        nPar_mean = np.zeros([len(outputs_nm),len(thrs)])
        pPar_mean = np.zeros([len(outputs_nm),len(thrs)])
        for j, iVar in enumerate(outputs_nm):
            nPar_mean[j,:] = dict_combinations[iVar][str(iPC)]['mean']['num_parents']
            pPar_mean[j,:] = dict_combinations[iVar][str(iPC)]['mean']['per_parents']
        nPar_mean, pPar_mean = np.mean(nPar_mean,axis=0), np.mean(pPar_mean,axis=0)
        to_be_updated = {str(iPC):{'thresholds':thrs,'num_parents':nPar_mean,'per_parents':pPar_mean,}}
        dict_combinations['mean'].update(to_be_updated)
#         dict_combinations['mean'][iPC] = {
#             'thresholds':thrs,
#             'num_parents':nPar_mean,
#             'per_parents':pPar_mean,
#         }
    
#     print()
#     print(dict_combinations['mean']['0.001']['num_parents'])
#     print(dict_combinations['mean']['0.1']['num_parents'])
#     import pdb; pdb.set_trace()
    
    return dict_combinations



# Plotting


def recommend_sizes(links):
    """
    
    """
    linked_variables = find_linked_variables(links)
    n_linked_vars = len(linked_variables)
    print(f"n_linked_vars : {n_linked_vars}")
    if n_linked_vars <= 35:
        # Small
        figsize = (16, 16)
        node_size = 0.15
    elif n_linked_vars <= 70:
        # Medium
        figsize = (32, 32)
        node_size = 0.10
    else:
        # Big
        figsize = (48, 48)
        node_size = 0.05

    return figsize, node_size


# def scale_val_matrix(val_matrix):
#     """
#     Scales to values between -1 and 1, keeping zero in the same place
#     """
#     max_val = np.abs(val_matrix).max()
#     return val_matrix/max_val


def scale_link_width(o_link_width, threshold):
    """
    Scales 0.05 (so links are drawn) and 1 range, taking the threshold
    as the minimum
    """
    min_val = threshold
    # min_val = o_link_width[o_link_width >= threshold].min()
    smallest_link = 0.05
    link_width = o_link_width + smallest_link
    link_width -= min_val
    link_width[o_link_width < threshold] = 0
    return link_width / link_width.max()


def plot_aggregated_results(var_names_parents, aggregated_results, setup):
    Path(setup.plots_folder).mkdir(parents=True, exist_ok=True)
    dict_combinations = build_plot_matrices(var_names_parents, aggregated_results)

    for combination, combination_results in dict_combinations.items():
        print(combination)
        plot_filename = "{cfg}_{combination}.png".format(
            cfg=Path(setup.yml_filename).name.rsplit(".")[0], combination=combination
        )
        plot_file = Path(setup.plots_folder, plot_filename)
        if not setup.overwrite_plots and plot_file.is_file():
            print(f"Found file {plot_file}, skipping.")
            continue  # Ignore this result
        links = combination_results["links"]
        figsize, node_size = recommend_sizes(links)

        var_names = combination_results["var_names"]

        val_matrix = combination_results["val_matrix"]
        edges_array = np.array([val_matrix.min(), val_matrix.max()])
        vmin_edges = np.max(np.abs(edges_array)) * -1.0
        vmax_edges = np.max(np.abs(edges_array))
        edge_ticks = (
            vmax_edges - vmin_edges
        ) * 0.2  # So it has the same proportions as default
        link_width = combination_results["link_width"]
        # val_matrix = scale_val_matrix(val_matrix)
        link_width = scale_link_width(link_width, float(combination.threshold))

        plot_links(
            links,
            var_names,
            val_matrix=val_matrix,
            vmin_edges=vmin_edges,
            vmax_edges=vmax_edges,
            edge_ticks=edge_ticks,
            link_width=link_width,
            arrow_linewidth=10,
            save_name=plot_file,
            figsize=figsize,
            node_size=node_size,
            show_colorbar=True,
        )

        
def plot_causal_metrics(
    aggregated_results, 
    setup, 
    save=False,
):
    dict_combinations = build_links_metrics(setup, aggregated_results)
    
    # Filenm format (folder and figure name)
    save_dir = setup.plots_folder
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    variables = [var.name for var in setup.list_spcam if var.type == "out"]
    variables = '-'.join(variables)
    pc_alphas  = [str(iPC) for iPC in setup.pc_alphas]
    pcs       = '-'.join(pc_alphas)
    lats = [str(iL) for iL in setup.region[0]];  lats = '-'.join(lats)
    lons = [str(iL) for iL in setup.region[-1]]; lons = '-'.join(lons)
    wgt_format = ['.png','_latwts.png'][setup.area_weighted]
    filenm = variables+'_a-'+pcs+'_lats-'+lats+'_lons'+lons+wgt_format
    if save: save = save_dir+'/'+filenm
    
    plot_links_metrics(setup, dict_combinations, save)