WeChat: cstutorcs
QQ: 749389476
Email: tutorcs@163.com
# pylint:disable=C0103

from collections import OrderedDict
import numpy as np
from ..base_model import BaseModel


class GABA_B(BaseModel):
    """GABA(B) Synapse Model"""

    Time_Scale = 1e3  # s to ms
    """GABA(B) Operates on Millisecond Scale"""
    Default_States = OrderedDict(x1=0.0, x2=0.0)
    """Default State Variables of the GABA(B) Model"""
    Default_Params = OrderedDict(
        alpha1=0.09,
        alpha2=0.18,
        beta1=0.0012,
        beta2=0.034,
        g_max=0.3,
        E=-90.0,
        n=4.0,
        gamma=100.0,
    )
    """Default Parameters of the GABA(B) Model"""

    def ode(self, t, states, I_ext=0.0):
        """GABA(B) gradient function"""
        x1, x2 = states
        d_x1 = I_ext * self.params["alpha1"] * (1 - x1) - self.params["beta1"] * x1
        d_x2 = self.params["alpha2"] * x1 - self.params["beta2"] * x2
        return [d_x1, d_x2]

    def conductance(self, x2):
        """GABA(B) conductance"""
        g = (
            self.params["g_max"]
            * x2 ** self.params["n"]
            / (self.params["gamma"] + x2 ** self.params["n"])
        )
        return g
