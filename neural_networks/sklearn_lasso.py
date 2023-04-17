from pathlib import Path
import xarray as xr
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import numpy as np
from pathlib import Path

from .cbrain.normalization import InputNormalizer, DictNormalizer, Normalizer
from .cbrain.utils import return_var_idxs, load_pickle

import sys; import sys; sys.path.append("..")
from utils.variable import Variable_Lev_Metadata

def sklasso(
    inputs,
    output,
    data_fn,
    norm_fn,
    setup,
):

    # Open datasets
    data_ds = xr.open_dataset(data_fn)
    norm_ds = xr.open_dataset(norm_fn)
    
    # Inputs (X) variables
    X_vars = inputs
    X_dict = vars_dict(X_vars)
    X_idx  = return_var_idxs(data_ds, X_dict)
    # print(f"X_idx: {X_idx}")
    
    # Output (Y) variable
    Y_var = output
    Y_dict = vars_dict([Y_var])
    Y_idx  = return_var_idxs(data_ds, Y_dict)
    # print(f"Y_idx: {Y_idx}")
    
    # Scaling
    XY = data_ds["vars"]
    XY_scaler=StandardScaler()
    XY_scaler.fit(XY)
    XY_Scale = XY_scaler.transform(XY)
    X_Scale = XY_Scale[:,X_idx]
    Y_Scale = XY_Scale[:,Y_idx]
    # print(f"X_Scale (data): {X_Scale.shape}")
    # print(f"Y_Scale (data): {Y_Scale.shape}")
            
    norm_ds.close()
    data_ds.close()
    
    # SKLEARN LASSO
    print(f"Computing Lasso regression...")
    lasso = Lasso(alpha=setup.alpha_lasso).fit(X_Scale,Y_Scale)
    # print(f"lasso.coef_: {lasso.coef_}")
    lasso_idx = [i for i in range(len(lasso.coef_)) if lasso.coef_[i] != -0.]
    lasso_inputs = [inputs[i] for i in lasso_idx]
    lasso_coefs = [[lasso.coef_[i],0.][lasso.coef_[i] == -0.] for i in range(len(lasso.coef_))]
    
    return lasso_inputs, lasso_coefs
    

def vars_dict(list_variables):
        """ Convert the given list of Variable_Lev_Metadata into a
        dictionary to be used on the data generator.
        
        Parameters
        ----------
        list_variables : list(Variable_Lev_Metadata)
            List of variables to be converted to the dictionary format
            used by the data generator
        
        Returns
        -------
        vars_dict : dict{str : list(int)}
            Dictionary of the form {ds_name : list of levels}, where
            "ds_name" is the name of the variable as stored in the
            dataset, and "list of levels" a list containing the indices
            of the levels of that variable to use, or None for 2D
            variables.
        """
        vars_dict = dict()
        for variable in list_variables:
            ds_name = variable.var.ds_name  # Name used in the dataset
            if variable.var.dimensions == 2:
                ds_name = 'LHF_nsDELQ' if ds_name == 'LHF_NSDELQ' else ds_name
                vars_dict[ds_name] = None
            elif variable.var.dimensions == 3:
                levels = vars_dict.get(ds_name, list())
                levels.append(variable.level_idx)
                vars_dict[ds_name] = levels
        return vars_dict
    
    