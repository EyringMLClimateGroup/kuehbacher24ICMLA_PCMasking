from neural_networks.model_diagnostics import ModelDiagnostics


def create_castle_model_description(setup, models):
    setup.model_type = setup.nn_type
    model_desc = ModelDiagnostics(setup=setup,
                                  models=models)

    return model_desc
