WeChat: cstutorcs
QQ: 749389476
Email: tutorcs@163.com
# pylint:disable=C0103

from collections import OrderedDict
import numpy as np
from ..base_model import BaseModel


class AMPA(BaseModel):
    """AMPA Synapse Model"""

    Time_Scale = 1e3  # s to ms
    """AMPA Operates on Millisecond Scale"""
    Default_States = OrderedDict(s=0.0)
    """Default State Variables of the AMPA Model"""
    Default_Params = OrderedDict(alpha=1.1, beta=0.19, g_max=0.05, E=0.0)
    """Default Parameters of the AMPA Model"""

    def ode(self, t, states, I_ext=0.0):
        """AMPA gradient function"""
        s = states
        d_s = I_ext * self.params["alpha"] * (1 - s) - self.params["beta"] * s
        return d_s

    def conductance(self, s):
        """AMPA conductance"""
        g = self.params["g_max"] * s
        return g
