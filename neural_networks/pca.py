import xarray as xr
from sklearn.decomposition import PCA
from joblib import dump, load
#from sklearn.preprocessing import StandardScaler

import numpy as np
from pathlib import Path

from .cbrain.utils import return_var_idxs


def pca(
    data_fn,
    pca_data_fn,
    input_vars_dict,
    norm_fn,
    load_pca_model,
    # save_dir,
    setup,
):
    
    # Open datasets
    data_ds = xr.open_dataset(data_fn)
    norm_ds = xr.open_dataset(norm_fn)
    
    # inputs idxs (all)
    var_idxs   = return_var_idxs(norm_ds, input_vars_dict)
    num_inputs = len(var_idxs)
    
    # Input (X) & output (Y) data
    X = data_ds["vars"][:,var_idxs]
    Y = data_ds["vars"][:,-(len(data_ds.var_names)-len(var_idxs)):]
    
    # Normalize
    print(f"... normalizing data...")
    sub, div = pca_norm(input_vars_dict, var_idxs, norm_ds, setup)
    X_norm = (X - sub) / div
    
    # PCA
    if load_pca_model:
        print(f"... loading pca model...")
        pca_fn = Path(setup.train_data_folder, setup.train_data_fn.split('.')[0]+f"_pca{setup.n_components}"+".joblib")
        pca = load(pca_fn)
    else:
        print(f"... creating pca model...")
        pca_fn = Path(
            setup.train_data_folder, 
            setup.train_data_fn.split('.')[0]+f"_pca{setup.n_components}"+".joblib"
        )
        n_components = float(setup.n_components) if setup.n_components < 1. else int(setup.n_components)
        pca = PCA(n_components=n_components)
        pca.fit(X_norm)
        print(f"... saving pca model: {pca_fn}...")
        dump(pca, pca_fn)
    PCs = pca.transform(X_norm)
    
    # Save data
    print(f"... saving PC-components...")
    # var_names = []; count = 1
    # for i in range(num_inputs):
    #     if count <= PCs.shape[-1]: 
    #         var_names.append('PC'+str(count))
    #     else:
    #         var_names.append('null')
    #     count += 1
    # var_concat = np.concatenate((np.array(var_names),data_ds.coords['var_names'][num_inputs:]))
    zero_inputs = np.zeros([len(PCs[:,0]),num_inputs-len(PCs[0])])
    vars_data = np.concatenate((PCs,zero_inputs,Y),axis=1)
    
    # define data with variable attributes
    data_vars = {
        'time':('time',np.array(data_ds.time)),
        'lat':('lat',np.array(data_ds.lat)),
        'lon':('lon',np.array(data_ds.lon)),
        'vars':(['sample','var_names'],vars_data),
                }
    coords = {
        # 'var_names': (['var_names'],var_concat),
        'var_names': (['var_names'],data_ds.coords['var_names'].data),
    }
    attrs = {'author':'Fernando Iglesias-Suarez', 
             'email':'fernando.iglesias-suarez@dlr.de',
             'Num. PCs':len(PCs[0]),
             'explained_variance_ratio_.cumsum':pca.explained_variance_ratio_.cumsum()}

    # create dataset
    pca_data_ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
    pca_data_ds.to_netcdf(pca_data_fn)
    print(f"... pca_data_ds: {pca_data_ds}.")
    
    norm_ds.close()
    data_ds.close()
    
    
def pca_norm(
    input_vars_dict,
    var_idxs,
    norm_ds,
    setup,
):
    
    div = setup.input_div
    sub = norm_ds[setup.input_sub].values[var_idxs]
    
    if div == "maxrs":
        rang = norm_ds["max"][var_idxs] - norm_ds["min"][var_idxs]
        std_by_var = rang.copy()
        for v in input_vars_dict.keys():
            std_by_var[std_by_var.var_names == v] = norm_ds["std_by_var"][
                norm_ds.var_names_single == v
            ]
        div = np.maximum(rang, std_by_var).values
    elif div == "std_by_var":
        # SR: Total mess. Should be handled better
        tmp_var_names = norm_ds.var_names[var_idxs]
        div = np.zeros(len(tmp_var_names))
        for v in input_vars_dict.keys():
            std_by_var = norm_ds["std_by_var"][norm_ds.var_names_single == v]
            div[tmp_var_names == v] = std_by_var
    else:
        div = norm_ds[div].values[var_idxs]
    
    return sub, div