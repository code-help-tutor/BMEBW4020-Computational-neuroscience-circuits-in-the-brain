WeChat: cstutorcs
QQ: 749389476
Email: tutorcs@163.com
# pylint:disable=C0103

from collections import OrderedDict
import numpy as np
from ..base_model import BaseModel


class SodiumChannelMemristor(BaseModel):
    """Sodium Channel Memristor Model"""

    Time_Scale = 1.0
    """Sodium Channel Operates on Second Scale"""
    Default_States = OrderedDict(M=1.0, H=2.3e-4)
    """Default State Variables of the Sodium Channel Memristor Model"""
    Default_Params = OrderedDict(V_0=115, G_0=120)
    """Default Parameters of the Sodium Channel Memristor Model"""

    def ode(self, t, states, I_ext=0.0):
        """Sodium channel gradient function"""
        M, H = states
        G = self.params["G_0"] * M**3 * H
        V = I_ext / G + self.params["V_0"]

        am = 0.1 * (25 - V) / (np.exp(2.5 - 0.1 * V) - 1)
        bm = 4 * np.exp(-V / 18)
        ah = 0.07 * np.exp(-V / 20)
        bh = 1 / (np.exp(3 - 0.1 * V) + 1)

        d_M = am * (1 - M) - bm * M
        d_H = ah * (1 - H) - bh * H
        return [d_M, d_H]
