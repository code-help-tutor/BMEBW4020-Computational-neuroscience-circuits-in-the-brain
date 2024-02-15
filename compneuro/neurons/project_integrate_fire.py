WeChat: cstutorcs
QQ: 749389476
Email: tutorcs@163.com
# pylint:disable=C0103

from collections import OrderedDict
import numpy as np
from tqdm.auto import tqdm  # pylint:disable
from .. import errors as err
from ..base_model import BaseModel


def pif_reset_callback(neuron: "ProjectIntegrateFire"):
    neuron.prc_ctr += 1
    spike_mask = neuron.states["V"] > neuron.params["V_T"]
    neuron.states["V"][spike_mask] = 0
    neuron.prc_ctr[spike_mask] = 0


class ProjectIntegrateFire(BaseModel):
    """Project Integrate and Fire (PIF) Neuron Model"""

    Time_Scale = 1
    """PIF Operates on Second Scale"""
    Default_States = OrderedDict(V=0.0)
    """Default State Variables of the PIF Model"""
    Default_Params = OrderedDict(V_T=1.0)
    """Default Parameters of the PIF Model"""
    callbacks = (pif_reset_callback,)
    """PIF Reset resets voltage and PRC counter"""
    Supported_Solvers = ("Euler",)
    """PIF only supports Euler since it needs callback"""

    def __init__(self, num: int = 1, callbacks=None, **kwargs):
        prc_V = kwargs.pop("prc_V", np.array([0.0]))
        super().__init__(num=num, callbacks=callbacks, **kwargs)
        self.prc_V = prc_V
        self.prc_ctr = np.zeros(num, dtype=int)

    def ode(self, t: float, states: np.ndarray, I_ext: float = None):
        """Definition of Differential Equation"""
        (V,) = states
        d_V = 1.0 + self.prc_V[self.prc_ctr % len(self.prc_V)] * I_ext

        return [d_V]
