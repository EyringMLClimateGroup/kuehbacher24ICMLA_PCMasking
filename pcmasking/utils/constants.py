from enum import Enum
from pathlib import Path
import os


class SPCAM_Vars(Enum):
    def __init__(self, dimensions, var_type, label):
        self._value_ = self._name_
        self.dimensions = dimensions
        self.type = var_type
        self.label = label
        self.ds_name = self._name_.upper()

    def __str__(self):
        return f"{self.__repr__()}, {self.label}"

    def __repr__(self):
        return f"({self._name_}, {self.dimensions}, {self.type})"

    @classmethod
    def _missing_(cls, value):
        for v in SPCAM_Vars:
            if value == v.name:
                return v
        raise ValueError(f"{value} is not a valid SPCAM_Vars")

    tphystndtdt = (3, "in", "Temperature tendency (t-1)")
    phqtdt = (3, "in", "Specific humidity tendency (t-1)")
    tbp = (3, "in", "Temperature")
    qbp = (3, "in", "Specific humidity")
    vbp = (3, "in", "Meridional wind")
    rh = (3, "in", "Relative humidity")
    bmse = (3, "in", "TODO")

    fsnttdt = (2, "in", "Net solar flux at top of model (t-1)")
    fsnstdt = (2, "in", "Net solar flux at surface (t-1)")
    flnttdt = (2, "in", "Net longwave flux at top of model (t-1)")
    flnstdt = (2, "in", "Net longwave flux at surface (t-1)")
    precttdt = (2, "in", "Precipitation (t-1)")
    ps = (2, "in", "Surf. Pressure")
    solin = (2, "in", "Incoming solar radiation")
    shflx = (2, "in", "Sensible heat flux")
    lhflx = (2, "in", "Latent heat flux")
    lhf_nsdelq = (2, "in", "TODO")

    tphystnd = (3, "out", "Temperature tendency")
    phq = (3, "out", "Specific humidity tendency")

    fsnt = (2, "out", "Net solar flux at top of model")
    fsns = (2, "out", "Net solar flux at surface")
    flnt = (2, "out", "Net longwave flux at top of model")
    flns = (2, "out", "Net longwave flux at surface")
    prect = (2, "out", "Precipitation")


ANCIL_FILE = os.path.join(Path(__file__).parent.parent.parent.resolve(), "data", "ancil_spcam.nc")
FILENAME_PATTERN = "{var_name}_{level}_{experiment}.nc"
AGGREGATE_PATTERN = "{var_name}_{ind_test_name}_{experiment}"
TAU_MIN = 1
TAU_MAX = 1
SIGNIFICANCE = "analytic"
