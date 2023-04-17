import xarray as xr
from sklearn.decomposition import PCA
from joblib import dump, load
from sklearn.preprocessing import StandardScaler

import numpy as np
from pathlib import Path

from .cbrain.utils import return_var_idxs


def pca(
    data_fn,
    pca_data_fn,
    input_vars_dict,
    norm_fn,
    load_pca_model,
    setup,
):
    
    # Open datasets
    data_ds = xr.open_dataset(data_fn)
    # print(f"data_fn: {data_fn}")
    norm_ds = xr.open_dataset(norm_fn)
    
    # inputs idxs (all)
    var_idxs   = return_var_idxs(norm_ds, input_vars_dict)
    # print(f"var_idxs: {var_idxs}")
    num_inputs = len(var_idxs)
    
    # Input (X) & output (Y) data
    X = data_ds["vars"][:,var_idxs]
    # print(f"X: {X}")
    Y = data_ds["vars"][:,-(len(data_ds.var_names)-len(var_idxs)):]
    # print(f"Y: {Y}")
    
    # Normalize
    print(f"... normalizing data...")
    X_scaler=StandardScaler()
    X_scaler.fit(X)
    X_norm = X_scaler.transform(X)
    # print(f"X_norm.shape: {X_norm.shape}")
    
    # PCA
    if load_pca_model:
        pca_fn = Path(setup.train_data_folder, setup.train_data_fn.split('.')[0]+"_pca"+".joblib")
        print(f"... loading pca model: {pca_fn}...")
        pca = load(pca_fn)
    else:
        pca_fn = Path(
            setup.train_data_folder, 
            # setup.train_data_fn.split('.')[0]+f"_pca{setup.n_components}"+".joblib"
            setup.train_data_fn.split('.')[0]+"_pca"+".joblib"
        )
        print(f"... creating pca model: {pca_fn}...")
        n_components = float(setup.n_components) if setup.n_components < 1. else int(setup.n_components)
        # pca = PCA(n_components=n_components)
        # print(f"num_inputs: {num_inputs}")
        pca = PCA(n_components=num_inputs)
        pca.fit(X_norm)
        print(f"... saving pca model: {pca_fn}...")
        dump(pca, pca_fn)
    PCs = pca.transform(X_norm)
    print(F"PCs.shape:{PCs.shape}")
    
    # Save data
    print(f"... saving PC-components...")
    # zero_inputs = np.zeros([len(PCs[:,0]),num_inputs-len(PCs[0])])
    # vars_data = np.concatenate((PCs,zero_inputs,Y),axis=1)
    vars_data = np.concatenate((PCs,Y),axis=1)
    # print(f"vars_data.shape: {vars_data.shape}")
    
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
             'explained_variance_ratio_.sum':pca.explained_variance_ratio_.sum(),
             'explained_variance_ratio_':pca.explained_variance_ratio_,
             'explained_variance_ratio_.cumsum':pca.explained_variance_ratio_.cumsum(),
             'explained_variance_':pca.explained_variance_
            }

    # create dataset
    pca_data_ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
    pca_data_ds.to_netcdf(pca_data_fn)
    print(f"... pca_data_ds: {pca_data_ds}.")
    
    norm_ds.close()
    data_ds.close()
