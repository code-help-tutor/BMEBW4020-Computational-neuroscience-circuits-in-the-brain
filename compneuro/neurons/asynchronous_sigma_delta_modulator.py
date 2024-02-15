WeChat: cstutorcs
QQ: 749389476
Email: tutorcs@163.com
# pylint:disable=C0103
from collections import OrderedDict
import numpy as np
from tqdm.auto import tqdm  # pylint:disable
from .. import errors as err
from ..base_model import BaseModel


def asdm_callback(neuron: "ASDM"):
    """Reset Neuron's Voltage to Baseline Value"""
    spike_mask = np.logical_and(
        np.abs(neuron.states["V"]) > neuron.params["delta"],
        np.sign(neuron.states["V"]) != np.sign(neuron.states["z"]),
    )
    neuron.states["z"][spike_mask] *= -1


class ASDM(BaseModel):
    """Asynchronous-Sigma/Delta-Modulator"""

    Time_Scale = 1
    """ASDM Operates on Second Scale"""
    Default_States = OrderedDict(V=0.0, z=-1.0)
    """Default State Variables of the ASDM Model"""
    Default_Params = OrderedDict(C=1.0, b=0.0, delta=1.0)
    """Default Parameters of the ASDM Model"""
    Supported_Solvers = ("Euler",)
    """ASDM only support Euler's method"""
    callbacks = (asdm_callback,)
    """ASDM flips threshold to baseline after spike"""

    def ode(self, t: float, states: np.ndarray, I_ext: float = None) -> np.ndarray:
        """Definition of Differential Equation"""
        (V, z) = states
        d_V = (I_ext - z * self.params["b"]) / self.params["C"]
        return [d_V, np.zeros_like(d_V)]
