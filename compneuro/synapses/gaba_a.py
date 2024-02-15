WeChat: cstutorcs
QQ: 749389476
Email: tutorcs@163.com
# pylint:disable=C0103

from collections import OrderedDict
import numpy as np
from ..base_model import BaseModel


class GABA_A(BaseModel):
    """GABA(A) Synapse Model"""

    Time_Scale = 1e3  # s to ms
    """GABA(A) Operates on Millisecond Scale"""
    Default_States = OrderedDict(s=0.0)
    """Default State Variables of the GABA(A) Model"""
    Default_Params = OrderedDict(alpha=5.0, beta=0.18, g_max=0.3, E=-70.0)
    """Default Parameters of the GABA(A) Model"""

    def ode(self, t, states, I_ext=0.0):
        """GABA(A) gradient function"""
        s = states
        d_s = I_ext * self.params["alpha"] * (1 - s) - self.params["beta"] * s
        return d_s

    def conductance(self, s):
        """GABA(A) conductance"""
        g = self.params["g_max"] * s
        return g
