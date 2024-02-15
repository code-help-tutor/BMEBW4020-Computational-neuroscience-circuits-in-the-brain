WeChat: cstutorcs
QQ: 749389476
Email: tutorcs@163.com
# pylint:disable=C0103
from collections import OrderedDict
import numpy as np
from ..base_model import BaseModel


class HodgkinHuxleyWilson(BaseModel):
    """Wilson Neuron Model

    The Wilson Neuron Model is a reduced version of the Hodgkin-Huxley Neuron
    model. Instead of having 4 state variables :math:`(V, n, m, h)`, wilson
    only has 2 state variables :math:`(V, R)`.
    """

    Time_Scale = 1e3
    """Wilson Operates on Millisecond Scale"""
    Default_States = OrderedDict(V=-70, R=0.088)
    """Default State Variables of the Wilson Model"""
    Default_Params = OrderedDict(C=1.2, E_K=-92.0, g_K=26.0, E_Na=55.0)
    """Default Parameters of the Wilson Model"""

    def ode(self, t, states, I_ext=0.0):
        """Wilson gradient function"""
        V, R = states

        ### BEGIN SOLUTION (tag:1)
        R_infty = 0.0135 * V + 1.03

        d_R = (R_infty - R) / 1.9

        I_Na = (17.81 + 0.4771 * V + 0.003263 * V**2) * (V - self.params["E_Na"])
        I_K = self.params["g_K"] * R * (V - self.params["E_K"])
        d_V = 1.0 / self.params["C"] * (I_ext - I_Na - I_K)
        ### END SOLUTION

        return [d_V, d_R]
