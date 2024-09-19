# Towards Physically Consistent Deep Learning for Climate Model Parameterizations

Developers: Birgit K&uuml;hbacher.   

Original [CI-NN](https://github.com/EyringMLClimateGroup/iglesias-suarez23jgr_CausalNNCAM)[^1] code by Fernando Iglesias-Suarez, Breixo Soliño Fernández.  
Original CBRAIN-CAM[^2] code by Stephan Rasp.  

This repository provides the source code used in the paper [Towards Physically Consistent Deep Learning for Climate Model Parameterization](https://arxiv.org/abs/2406.03920) by K&uuml;hbacher et al. 

[//]: # (Corresponding DOI: todo)

## Installation

To install the dependencies, it is recommended to use Anaconda or Conda. An environment file is provided in `dependencies.yml`.

## How to reproduce the results

The results described in the paper where obtained by executing the following steps:

1. Specify network and training configuration in configuration files for PreMaskNet (pre-masking phase) and MaskNet (masking phase). <br>Example configuration files can be found in `config`. 
    <p> <br> </p>
2. Run [main_train_pcmasking.py](main_train_pcmasking.py) for PreMaskNet. <br>Usage:`$ python main_train_pcmasking.py -c config.yml -s 42`
    * [main_train_pcmasking.py](main_train_pcmasking.py) trains all 65 models. It is possible to train just a subset of models by using [main_train_pcmasking_subset.py](main_train_pcmasking_subset.py). <br>Usage: `$ python main_train_pcmasking_subset.py -c config.yml -i inputs.txt -o outputs.txt -x "1-10" -s 42` <br>This call will train only networks for the variables with indices 1-10 in `outputs.txt`.
    * The lists of input and output variables can be generated with [main_generate_inputs_outputs_lists.py](main_generate_inputs_outputs_lists.py). <br>Usage: `$ python create_inputs_outputs.py -c config.yml`
    <p> <br> </p>
3. (Optional) Update the masking vector directory in the MaskNet configuration. 
    <p> <br> </p>
4. Run [main_train_mask_net_thresholds.py](main_train_mask_net_thresholds.py) to train MaskNet for multiple thresholds based on the masking vector values. <br>Usage: `$ python main_train_mask_net_thresholds.py -c config.yml -i inputs.txt -o outputs.txt -x 0 -r 75 -f fine_tune_cfg.yml -s 42` <br>This call will train only the variable with index 0 in `outputs.txt` for 20 threshold values between 0.001 and the 75th percentile of the values in the masking vector. <br>The model weights are reloaded from `fine_tune_cfg.yml`.  
    * If training is to be done with a single threshold, use [main_train_pcmasking.py](main_train_pcmasking.py) or [main_train_pcmasking_subset.py](main_train_pcmasking_subset.py). 
    <p> <br> </p>
5. Evaluate trained PCMasking networks.
    * To compute SHAP values: [main_compute_shap_values.py](pcmasking%2Foffline_evaluation%2Fmain_compute_shap_values.py). <br>Usage: `$ python main_compute_shap_values.py -c config.yml -o outputs.txt -m map.txt --plot_directory ./output --n_time 1440 --n_samples 1000 --metric abs_mean` 
    * To plot SHAP values: Run the notebook [offline_evaluation_shap.ipynb](notebooks%2Foffline_evaluation_shap.ipynb)
    * To compute vertical cross-sections (including $R^2$): [main_plot_cross_section.py](pcmasking%2Foffline_evaluation%2Fmain_plot_cross_section.py). <br>Usage: `$ python main_plot_cross_section.py -c config.yml --plot_directory ./output`
    * To compute vertical profiles (including $R^2$): [main_plot_profiles.py](pcmasking%2Foffline_evaluation%2Fmain_plot_profiles.py). <br>Usage: `$ python main_plot_profiles.py -c config.yml --plot_directory ./output`
    * To plot vertical cross-sections and profiles (including $R^2$): Run the notebook [offline_evaluation_profile_cross_section_r2.ipynb](notebooks%2Foffline_evaluation_profile_cross_section_r2.ipynb).
    * To plot physical drivers: Run the notebook [offline_evaluation_physical_drivers.ipynb](notebooks%2Foffline_evaluation_physical_drivers.ipynb). 
   
<p> <br><br> </p>

[^1]: F. Iglesias-Suarez et al., “Causally-Informed Deep Learning to Improve Climate Models and Projections,” Journal of Geophysical Research: Atmospheres, vol. 129, no. 4, 2024, doi: 10/gtmfpk.  
[^2]: S. Rasp, M. S. Pritchard, and P. Gentine, “Deep learning to represent subgrid processes in climate models,” Proc. Natl. Acad. Sci. U.S.A., vol. 115, no. 39, pp. 9684–9689, Sep. 2018, doi: 10/gfcpxb.
