WeChat: cstutorcs
QQ: 749389476
Email: tutorcs@163.com
# pylint:disable=C0103

from collections import OrderedDict
import numpy as np
from ..base_model import BaseModel


class NMDA(BaseModel):
    """NMDA Synapse Model"""

    Time_Scale = 1e3  # s to ms
    """NMDA Operates on Millisecond Scale"""
    Default_States = OrderedDict(s=0.0)
    """Default State Variables of the NMDA Model"""
    Default_Params = OrderedDict(alpha=0.072, beta=0.0066, g_max=5.0, E=0.0)
    """Default Parameters of the NMDA Model"""

    def ode(self, t, states, I_ext=0.0):
        """NMDA gradient function"""
        s = states
        d_s = I_ext * self.params["alpha"] * (1 - s) - self.params["beta"] * s
        return d_s

    def conductance(self, s, V, Mg):
        """NMDA conductance"""
        E_1_2 = 16.13 * np.log(Mg / 3.57)
        G_NMDA = 1 / (1 + np.exp(-(V - E_1_2) / 16.13))
        g = G_NMDA * self.params["g_max"] * s
        return g


class NMDA_v2(BaseModel):
    """NMDA(v2) Synapse Model"""

    Time_Scale = 1e3  # s to ms
    """NMDA(v2) Operates on Millisecond Scale"""
    Default_States = OrderedDict(x1=0.0, x2=0.0)
    """Default State Variables of the NMDA(v2) Model"""
    Default_Params = OrderedDict(
        alpha1=0.072, alpha2=0.18, beta1=0.0066, beta2=0.034, g_max=4.5, E=0.0
    )
    """Default Parameters of the NMDA(v2) Model"""

    def ode(self, t, states, I_ext=0.0):
        """NMDA(v2) gradient function"""
        x1, x2 = states
        d_x1 = I_ext * self.params["alpha1"] * (1 - x1) - self.params["beta1"] * x1
        d_x2 = x1 * self.params["alpha2"] * (1 - x2) - self.params["beta2"] * x2
        return [d_x1, d_x2]

    def conductance(self, x2, V, Mg):
        """NMDA conductance"""
        E_1_2 = 16.13 * np.log(Mg / 3.57)
        G_NMDA = 1 / (1 + np.exp(-(V - E_1_2) / 16.13))
        g = G_NMDA * self.params["g_max"] * x2
        return g
