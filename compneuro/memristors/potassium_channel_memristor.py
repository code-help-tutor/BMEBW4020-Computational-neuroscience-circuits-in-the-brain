WeChat: cstutorcs
QQ: 749389476
Email: tutorcs@163.com
# pylint:disable=C0103

from collections import OrderedDict
import numpy as np
from ..base_model import BaseModel


class PotassiumChannelMemristor(BaseModel):
    """Potassium Channel Memristor Model"""

    Time_Scale = 1.0
    """Potassium Channel Operates on Second Scale"""
    Default_States = OrderedDict(N=0.16)
    """Default State Variables of the Potassium Channel Memristor Model"""
    Default_Params = OrderedDict(V_0=-12, G_0=36)
    """Default Parameters of the Potassium Channel Memristor Model"""

    def ode(self, t, states, I_ext=0.0):
        """Potassium channel gradient function"""
        N = states
        G = self.params["G_0"] * N**4
        V = I_ext / G + self.params["V_0"]

        an = 0.01 * (10 - V) / (np.exp(1 - 0.1 * V) - 1)
        bn = 0.125 * np.exp(-V / 80)

        d_N = an * (1 - N) - bn * N
        return d_N
