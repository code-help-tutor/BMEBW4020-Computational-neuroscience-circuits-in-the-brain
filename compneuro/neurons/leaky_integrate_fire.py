WeChat: cstutorcs
QQ: 749389476
Email: tutorcs@163.com
# pylint:disable=C0103

from collections import OrderedDict
import numpy as np
from tqdm.auto import tqdm  # pylint:disable
from .. import errors as err
from ..base_model import BaseModel


def spike_reset_callback(neuron: "LeakyIntegrateFire"):
    """Reset Neuron's Voltage to Baseline Value"""
    spike_mask = neuron.states["V"] > neuron.params["V_T"]
    neuron.states["V"][spike_mask] = neuron.params["V_0"]


class LeakyIntegrateFire(BaseModel):
    """Leaky Integrate and Fire (LIF) Neuron Model"""

    Time_Scale = 1
    """LIF Operates on Second Scale"""
    Default_States = OrderedDict(V=0.0)
    """Default State Variables of the LIF Model"""
    Default_Params = OrderedDict(C=1.0, R=1.0, V_0=0.0, V_T=1.0)
    """Default Parameters of the LIF Model"""
    Supported_Solvers = ("Euler",)
    """LIF only support Euler's method"""
    callbacks = (spike_reset_callback,)
    """LIF resets voltage to baseline after spike"""

    def ode(self, t: float, states: np.ndarray, I_ext: float = None) -> np.ndarray:
        """Definition of Differential Equation"""
        (V,) = states
        d_V = (I_ext - V / self.params["R"]) / self.params["C"]
        return [d_V]
