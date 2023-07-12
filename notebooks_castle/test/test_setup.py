import logging
import unittest

import yaml

from utils.setup import SetupNeuralNetworks

class TestCastleSetup(unittest.TestCase):
    logging.basicConfig(level=logging.INFO)

    def test_castle_setup(self):
        logging.info("Testing creating setup instance for CASTLE configuration.")

        argv = ["-c", "cfg_castle_NN_Creation_test_1.yml"]

        castle_setup = SetupNeuralNetworks(argv)

        with open(argv[1], "r") as yml_cfg_file:
            yml_cfg = yaml.load(yml_cfg_file, Loader=yaml.FullLoader)

        self.assertEqual(castle_setup.nn_type, "castleNN")
        self.assertTrue(castle_setup.do_castle_nn)

        self.assertEqual(castle_setup.rho, yml_cfg["rho"])
        self.assertEqual(castle_setup.alpha, yml_cfg["alpha"])
        self.assertEqual(castle_setup.beta, yml_cfg["beta"])
        self.assertEqual(castle_setup.lambda_, yml_cfg["lambda"])