import pytest

from neural_networks.load_models import load_model_weights_from_checkpoint, \
    load_model_from_previous_training
from neural_networks.models import generate_models
from neural_networks.training import train_all_models
from test.testing_utils import set_memory_growth_gpu, train_model_if_not_exists, \
    build_test_gen, set_strategy

try:
    set_memory_growth_gpu()
except RuntimeError:
    print("\n\n*** GPU growth could not be enabled. "
          "When running multiple tests, this may be due physical drivers having already been "
          "initialized, in which case memory growth may already be enabled. "
          "If memory growth is not enabled, the tests may fail with CUDA error. ***\n")


@pytest.mark.parametrize("strategy", ["", "mirrored"])
@pytest.mark.parametrize("setup_str", ["setup_castle_adapted_2d_dagma", "setup_castle_adapted_2d_notears",
                                       "setup_castle_original_2d", ])
def test_load_model_weights_from_checkpoint_castle_model_description(setup_str, strategy, request):
    setup = request.getfixturevalue(setup_str)
    setup = set_strategy(setup, strategy)

    train_model_if_not_exists(setup)
    model_descriptions = generate_models(setup)

    for md in model_descriptions:
        # Evaluate the model
        test_gen = build_test_gen(md, setup)
        print(f"\nEvaluating untrained model {md}.")
        with test_gen:
            md.model.evaluate(test_gen, verbose=2)

        md = load_model_weights_from_checkpoint(md, which_checkpoint="best")

        print(f"\nEvaluating model {md} with loaded weights from checkpoint.")
        with test_gen:
            md.model.evaluate(test_gen, verbose=2)


@pytest.mark.parametrize("strategy", ["", "mirrored"])
@pytest.mark.parametrize("setup_str", ["setup_castle_adapted_2d_dagma", "setup_castle_adapted_2d_notears",
                                       "setup_castle_original_2d", ])
def test_train_castle_model_description_load_from_ckpt_true(setup_str, strategy, request):
    setup = request.getfixturevalue(setup_str)
    setup = set_strategy(setup, strategy)

    train_model_if_not_exists(setup)

    # Train from checkpoint
    model_descriptions = generate_models(setup)
    train_all_models(model_descriptions, setup, from_checkpoint=True)


@pytest.mark.parametrize("strategy", ["", "mirrored"])
@pytest.mark.parametrize("setup_str", ["setup_castle_adapted_2d_dagma", "setup_castle_adapted_2d_notears",
                                       "setup_castle_original_2d", ])
def test_load_model_from_previous_training_castle_model_description(setup_str, strategy, request):
    setup = request.getfixturevalue(setup_str)
    setup = set_strategy(setup, strategy)

    train_model_if_not_exists(setup)
    model_descriptions = generate_models(setup)

    for md in model_descriptions:
        # Evaluate the model
        test_gen = build_test_gen(md, setup)
        print(f"\nEvaluated untrained model {md}.")
        with test_gen:
            md.model.evaluate(test_gen, verbose=2)

        md.model = load_model_from_previous_training(md)

        print(f"\nEvaluated model {md} with loaded weights.")
        with test_gen:
            md.model.evaluate(test_gen, verbose=2)


@pytest.mark.parametrize("strategy", ["", "mirrored"])
@pytest.mark.parametrize("setup_str", ["setup_castle_adapted_2d_dagma", "setup_castle_adapted_2d_notears",
                                       "setup_castle_original_2d", ])
def test_train_castle_model_description_continue_training_true(setup_str, strategy, request):
    setup = request.getfixturevalue(setup_str)
    setup = set_strategy(setup, strategy)

    train_model_if_not_exists(setup)

    model_descriptions = generate_models(setup, continue_training=True)
    train_all_models(model_descriptions, setup, continue_training=True)
