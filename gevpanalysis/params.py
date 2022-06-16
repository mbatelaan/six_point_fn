from fractions import Fraction
import numpy as np
import yaml


def recursive_cast(value, dtype):
    try:
        if isinstance(value, str):
            raise TypeError
        return [recursive_cast(item, dtype) for item in value]
    except TypeError:
        return dtype(value)


class params:
    def __init__(self, kappas):
        files = {0: "kp121040kp120620.yaml"}

        try:
            ConfigFile = files[kappas]
        except KeyError:
            raise ValueError(f"Sorry, no config file for kappa={kappas}")

        with open(ConfigFile) as f:
            config = yaml.safe_load(f)

        # Set things to defaults defined in another YAML file
        with open("defaults.yaml") as f:
            defaults = yaml.safe_load(f)
        for key, value in defaults.items():
            config.setdefault(key, value)

        # Cast things as arrays that need to be
        array_variables = ["bounds2pt", "boundsratio", "norma"]
        for var in array_variables:
            array = config[var]
            array = recursive_cast(array, float)
            config[var] = np.array(array)

        # Cast things as arrays that need to be (integer)
        array_variables = ["qval"]
        for var in array_variables:
            array = config[var]
            config[var] = np.array(array)

        # config["operators2"][2] = np.array(config["operators2"][2])
        # config["operators2"][8] = np.array(config["operators2"][8])

        # # Cast things as arrays that need to be (Fractions)
        # array_variables = ["baryons"]
        # for var in array_variables:
        #     array = config[var]
        #     array = recursive_cast(array, Fraction)
        #     array = recursive_cast(array, float)
        #     config[var] = np.array(array)

        # Cast things as arrays that need to be (Fractions)
        for var in config["baryons"]:
            array = var["charge"]
            array = recursive_cast(array, Fraction)
            array = recursive_cast(array, float)
            var["charge"] = np.array(array)

        config["geom"] = str(config["L"]) + "x" + str(config["T"])

        # self.__dict__ = config
        self.__dict__.update(config)
