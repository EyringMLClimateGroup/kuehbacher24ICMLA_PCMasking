# import os
# from pathlib import Path
# from neural_networks.load_models import get_path
# eval_cases  = ['0k'] # ['0k', 'm4k', 'p4k']
# statsnm     = 'shap'
# metric      = 'abs_mean' # 'mean', 'abs_mean', 'abs_mean_sign'
# nTime       = 1440 # 1440 # ~1-month
# nSamples    = 4096 # 1024; 2048; 4096; 8192 # int(setup.batch_size*8) # batch_size (1024) * 8 = SPCAM grid (128*64=8192)
#
# def compute_shapley_dict(model_desc, model_type, setup, out_path, outFile_dict_str, n_time, n_samples, metric):
#
#     pathnm = get_path(setup, setup.nn_type)
#
#     if not os.path.exists(pathnm):
#         Path(pathnm).mkdir(parents=True, exist_ok=True)
#
#     eval_dict = get_dict(pathnm, outFile)
#     if iCase not in eval_dict.keys(): eval_dict[iCase] = {}
#
#     input_list = get_var_list(setup, setup.spcam_inputs)
#     setup.inputs = sorted(
#         [Variable_Lev_Metadata.parse_var_name(p) for p in input_list],
#         key=lambda x: setup.input_order_list.index(x),
#     )
#
#         print('')
#
#         for iMod, model_type in enumerate(models_type):
#             print(model_type)
#             if model_type == 'SingleNN':
#
#                 md = ModelDiagnostics(setup=setup, models=models[model_type])
#
#                 if 'SingleNN' not in eval_dict[iCase].keys(): eval_dict[iCase][model_type] = {}
#                 for var in models[model_type].keys():
#
#                     if var.var.value not in eval_dict[iCase][model_type].keys():
#                         eval_dict[iCase][model_type][var.var.value] = {}
#                         shape = ma.masked_equal(ma.zeros([30, len(setup.inputs)]), 0.)
#                         eval_dict[iCase][model_type][var.var.value] = {
#                             statsnm: (statsnm, shape),
#                             'lab': model_type,
#                         }
#                         # save_dict(eval_dict, pathnm, outFile, verbose=1)
#
#                     i_level = var.level_idx if var.level_idx != None else 29
#                     if ma.is_masked(ma.mean(eval_dict[iCase][model_type][var.var.value][statsnm][1][i_level])):
#                         print(f'evaluate: {var}')
#
#                         # model, X_train, X_test, input_vars_dict = md.get_shapley_values('range', var, nTime=nTime)
#                         shap_values_mean, inputs, input_vars_dict = md.get_shapley_values('range', var, nTime=nTime,
#                                                                                           nSamples=nSamples,
#                                                                                           metric=metric)
#                         eval_dict[iCase][model_type][var.var.value][statsnm][1][i_level] = shap_values_mean
#
#                         # Save to pickle object
#                         save_dict(eval_dict, pathnm, outFile)
#
#                         print()
#
#                 print('\n \n')
#
#
#             elif model_type == 'CausalSingleNN':
#                 for pc_alpha in models[model_type].keys():
#                     for threshold in models[model_type][pc_alpha].keys():
#                         print(f"pc_alpha-threshold: {pc_alpha}-{threshold}")
#
#                         md = ModelDiagnostics(setup=setup, models=models[model_type][pc_alpha][threshold])
#
#                         c_model_type = f'{model_type}_{pc_alpha}_{threshold}'
#                         print(f'c_model_type: {c_model_type}')
#                         print('here')
#                         if c_model_type not in eval_dict[iCase].keys(): eval_dict[iCase][c_model_type] = {}
#                         for var in models[model_type][pc_alpha][threshold].keys():
#
#                             if var.var.value not in eval_dict[iCase][c_model_type].keys():
#                                 eval_dict[iCase][c_model_type][var.var.value] = {}
#                                 shape = ma.masked_equal(ma.zeros([30, len(setup.inputs)]), 0.)
#                                 eval_dict[iCase][c_model_type][var.var.value] = {
#                                     statsnm: (statsnm, shape),
#                                     'lab': model_type,
#                                 }
#
#                             i_level = var.level_idx if var.level_idx != None else 29
#                             if ma.is_masked(
#                                     ma.mean(eval_dict[iCase][c_model_type][var.var.value][statsnm][1][i_level])):
#                                 print(f'evaluate: {var}')
#
#                                 shap_values_mean, inputs, input_vars_dict = md.get_shapley_values('range', var,
#                                                                                                   nTime=nTime,
#                                                                                                   nSamples=nSamples,
#                                                                                                   metric=metric)
#                                 eval_dict[iCase][c_model_type][var.var.value][statsnm][1][i_level][
#                                     inputs] = shap_values_mean
#
#                                 # Save to pickle object
#                                 save_dict(eval_dict, pathnm, outFile)
#
#                                 print()
#
#                         print('\n \n')
#
#         print('\n \n \n')
#
#     return eval_dict