import sys, time, datetime
from datetime import datetime as dt
import numpy as np
from pcmasking.utils import utils as utils
from pcmasking.utils.pcmci_algorithm import find_links, pearsonr


def proc_analysis(
    analysis,
    gridpoints,
    var_parents,
    var_children,
    cond_ind_test,
    ind_test_name,
    pc_alphas,
    shifting,
    levels,
    parents_idx_levs,
    children_idx_levs,
    idx_lats,
    idx_lons,
    data_folder,
    experiment,
    output_file_pattern,
    output_folder,
    overwrite,
    verbosity,
):
    if analysis == "single":
        print("Analysis: single")
        print()
        single(
            gridpoints,
            var_parents,
            var_children,
            cond_ind_test,
            ind_test_name,
            pc_alphas,
            shifting,
            levels,
            parents_idx_levs,
            children_idx_levs,
            idx_lats,
            idx_lons,
            data_folder,
            experiment,
            output_file_pattern,
            output_folder,
            overwrite,
            verbosity,
        )
    elif analysis == "concat":
        print("Analysis: concat")
        print()
        concat(
            gridpoints,
            var_parents,
            var_children,
            cond_ind_test,
            ind_test_name,
            pc_alphas,
            levels,
            parents_idx_levs,
            children_idx_levs,
            idx_lats,
            idx_lons,
            data_folder,
            experiment,
            output_file_pattern,
            output_folder,
            overwrite,
            verbosity,
        )
    else:
        raise TypeError(
            "Please specify a valid analysis, i.e., 'single' or 'concat'; stop script"
        )


def single(
    gridpoints,
    var_parents,
    var_children,
    cond_ind_test,
    ind_test_name,
    pc_alphas,
    shifting,
    levels,
    parents_idx_levs,
    children_idx_levs,
    idx_lats,
    idx_lons,
    data_folder,
    experiment,
    output_file_pattern,
    output_folder,
    overwrite,
    verbosity,
):

    ## Processing
    len_grid = len(gridpoints)
    t_start = time.time()
    for i_grid, (lat, lon) in enumerate(gridpoints):

        t_start_gridpoint = time.time()
        data_parents = None

        idx_lat = idx_lats[i_grid]
        idx_lon = idx_lons[i_grid]

        for child in var_children:
            print(f"{dt.now()} Variable: {child.name}")
            if child.dimensions == 2:
                child_levels = [[levels[-1], 0]]
            elif child.dimensions == 3:
                child_levels = children_idx_levs

            for level in child_levels:
                results_file = utils.generate_results_filename_single(
                    child,
                    level[-1],
                    lat,
                    lon,
                    ind_test_name,
                    experiment,
                    output_file_pattern,
                    output_folder,
                )

                if not overwrite and results_file.is_file():
                    print(f"{dt.now()} Found file {results_file}, skipping.")
                    continue  # Ignore this level

                # Only load parents if necessary to analyze a child
                # they stay loaded until the next gridpoint
                if data_parents is None:

                    print(
                        f"{dt.now()} Gridpoint {i_grid+1}/{len_grid}:"
                        f" lat={lat} ({idx_lat}), lon={lon} ({idx_lon})"
                    )

                    print(f"Load Parents (state fields)...")
                    t_before_load_parents = time.time()
                    data_parents = utils.load_data(
                        var_parents,
                        0, # shifting always 0 (no-shifting)
                        experiment,
                        data_folder,
                        parents_idx_levs,
                        idx_lat,
                        idx_lon,
                    )
                    time_load_parents = datetime.timedelta(
                        seconds=time.time() - t_before_load_parents
                    )
                    print(f"{dt.now()} All parents loaded. Time: {time_load_parents}")

                # Process child
                data_child = utils.load_data(
                    [child], shifting, experiment, data_folder, [level], idx_lat, idx_lon
                )
                data = [*data_parents, *data_child]

                # Find links
                print(
                    f"{dt.now()} "
                    f"Finding links for {child.name} at level {level[-1]+1}"
                )
                t_before_find_links = time.time()
                if ind_test_name == 'parcorr':
                    results = find_links(data, pc_alphas, cond_ind_test, verbosity)
                elif ind_test_name == 'pearsonr':
                    results = pearsonr(data, pc_alphas, cond_ind_test, verbosity)
                else:
                    print(f"Invalid independence_test: {setup.independence_test}; stop!")
                    sys.exit()
                time_links = datetime.timedelta(
                    seconds=time.time() - t_before_find_links
                )
                total_time = datetime.timedelta(seconds=time.time() - t_start)
                print(
                    f"{dt.now()} Links found. Time: {time_links}"
                    f" Total time so far: {total_time}"
                )

                # Store causal links
                utils.save_results(results, results_file)

        time_point = datetime.timedelta(seconds=time.time() - t_start_gridpoint)
        total_time = datetime.timedelta(seconds=time.time() - t_start)
        print(
            f"{dt.now()} All links in gridpoint found. Time: {time_point}."
            f" Total time so far: {total_time}"
        )
        print()

    print(f"{dt.now()} Execution complete. Total time: {total_time}")


def concat(
    gridpoints,
    var_parents,
    var_children,
    cond_ind_test,
    ind_test_name,
    pc_alphas,
    levels,
    parents_idx_levs,
    children_idx_levs,
    idx_lats,
    idx_lons,
    data_folder,
    experiment,
    output_file_pattern,
    output_folder,
    overwrite,
    verbosity,
):

    ## Processing
    len_grid = len(gridpoints)
    t_start = time.time()
    data_parents = None

    ## outFile exists?
    for child in var_children:
        print(f"{dt.now()} Variable: {child.name}")
        if child.dimensions == 2:
            child_levels = [[levels[-1], 0]]
        elif child.dimensions == 3:
            child_levels = children_idx_levs
        for level in child_levels:

            results_file = utils.generate_results_filename_concat(
                child,
                level[-1],
                gridpoints,
                ind_test_name,
                experiment,
                output_file_pattern,
                output_folder,
            )

            if not overwrite and results_file.is_file():
                print(f"{dt.now()} Found file {results_file}, skipping.")
                continue  # Ignore this level

            # Only load parents if necessary to analyze a child
            # they stay loaded until the next gridpoint
            if data_parents is None:
                print()
                print(f"Load Parents (state fields)...")
                t_before_load_parents = time.time()
                for i_grid, (lat, lon) in enumerate(gridpoints):

                    t_start_gridpoint = time.time()

                    idx_lat = idx_lats[i_grid]
                    idx_lon = idx_lons[i_grid]

                    print(
                        f"{dt.now()} Gridpoint {i_grid+1}/{len_grid}:"
                        f" lat={lat} ({idx_lat}), lon={lon} ({idx_lon})"
                    )

                    normalized_parents = utils.load_data_concat(
                        var_parents,
                        experiment,
                        data_folder,
                        parents_idx_levs,
                        idx_lat,
                        idx_lon,
                    )
                    if data_parents is None:
                        data_parents = normalized_parents
                    else:
                        data_parents = np.concatenate(
                            (data_parents, normalized_parents), axis=1
                        )
                print("Parents shape: ", data_parents.shape)
                # Format data
                data_parents = utils.format_data(
                    data_parents, var_parents, parents_idx_levs
                )

                time_load_parents = datetime.timedelta(
                    seconds=time.time() - t_before_load_parents
                )
                print(f"{dt.now()} All parents loaded. Time: {time_load_parents}")
                print()

            # Process data child
            print(f"Load {child.name}...")
            t_before_load_child = time.time()
            data_child = None
            for i_grid, (lat, lon) in enumerate(gridpoints):

                idx_lat = idx_lats[i_grid]
                idx_lon = idx_lons[i_grid]

                print(
                    f"{dt.now()} Gridpoint {i_grid+1}/{len_grid}:"
                    f" lat={lat} ({idx_lat}), lon={lon} ({idx_lon})"
                )

                normalized_child = utils.load_data_concat(
                    [child], experiment, data_folder, [level], idx_lat, idx_lon
                )
                if data_child is None:
                    data_child = normalized_child
                else:
                    data_child = np.concatenate((data_child, normalized_child), axis=1)
            print("Child shape: ", data_child.shape)
            time_load_child = datetime.timedelta(
                seconds=time.time() - t_before_load_child
            )
            print(f"{dt.now()} Child loaded. Time: {time_load_child}")
            print()

            # Format data
            data_child = utils.format_data(data_child, [child], [level])
            data = [*data_parents, *data_child]

            # Find links
            print(f"{dt.now()} Finding links for {child.name} at level {level[-1]+1}")
            t_before_find_links = time.time()
            results = find_links(data, pc_alphas, cond_ind_test, verbosity)
            time_links = datetime.timedelta(seconds=time.time() - t_before_find_links)
            total_time = datetime.timedelta(seconds=time.time() - t_start)
            print(
                f"{dt.now()} Links found. Time: {time_links}"
                f" Total time so far: {total_time}"
            )
            print()

            # Store causal links
            utils.save_results(results, results_file)

    total_time = datetime.timedelta(seconds=time.time() - t_start)
    print(f"{dt.now()} Execution complete. Total time: {total_time}")
