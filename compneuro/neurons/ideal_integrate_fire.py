WeChat: cstutorcs
QQ: 749389476
Email: tutorcs@163.com
# pylint:disable=C0103

from collections import OrderedDict
import numpy as np
from tqdm.auto import tqdm  # pylint:disable
from .. import errors as err
from ..base_model import BaseModel


def spike_reset_callback(neuron: "IdealIntegrateFire"):
    """Reset Neuron's Voltage to Baseline Value"""
    spike_mask = neuron.states["V"] > neuron.params["V_T"]
    neuron.states["V"][spike_mask] = neuron.params["V_0"]


class IdealIntegrateFire(BaseModel):
    """Ideal Integrate and Fire (IAF) Neuron Model"""

    Time_Scale = 1
    """IAF Operates on Second Scale"""
    Default_States = OrderedDict(V=0.0)
    """Default State Variables of the IAF Model"""
    Default_Params = OrderedDict(C=1.0, V_0=0.0, V_T=1.0)
    """Default Parameters of the IAF Model"""
    Supported_Solvers = ("Euler",)
    """IAF only support Euler's method"""
    callbacks = (spike_reset_callback,)
    """IAF resets voltage to baseline after spike"""

    def ode(self, t: float, states: np.ndarray, I_ext: float = None) -> np.ndarray:
        """Definition of Differential Equation"""
        (V,) = states
        d_V = I_ext / self.params["C"]
        return [d_V]
