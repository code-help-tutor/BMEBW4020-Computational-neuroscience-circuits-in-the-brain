WeChat: cstutorcs
QQ: 749389476
Email: tutorcs@163.com
# pylint:disable=C0103

from collections import OrderedDict
import numpy as np
from ..base_model import BaseModel


class HodgkinHuxley(BaseModel):
    """Hodgkin Huxley Neuron Model"""

    Time_Scale = 1e3  # s to ms
    """Hodgkin-Huxley Operates on Millisecond Scale"""
    Default_States = OrderedDict(
        V=-60, n=(0.0, 0.0, 1.0), m=(0.0, 0.0, 1.0), h=(1.0, 0.0, 1.0)
    )
    """Default State Variables of the Hodgkin-Huxley Model"""
    Default_Params = OrderedDict(
        g_Na=120.0, g_K=36.0, g_L=0.3, E_Na=50.0, E_K=-77.0, E_L=-54.387
    )
    """Default Parameters of the Hodgkin-Huxley Model"""

    def ode(self, t, states, I_ext=0.0):
        """Hodgkin-Huxley gradient function"""
        V, n, m, h = states

        alpha = np.exp(-(V + 55.0) / 10.0) - 1.0
        beta = 0.125 * np.exp(-(V + 65.0) / 80.0)
        d_n_normal = (-0.01 * (V + 55.0) / alpha) * (1.0 - n) - beta * n
        _mask = abs(alpha) <= 1e-7
        d_n_small = 0.1 * (1.0 - n) - beta * n
        d_n = d_n_normal * (1 - _mask) + d_n_small * _mask

        alpha = np.exp(-(V + 40.0) / 10.0) - 1.0
        beta = 4.0 * np.exp(-(V + 65.0) / 18.0)
        d_m_normal = (-0.1 * (V + 40.0) / alpha) * (1.0 - m) - beta * m
        _mask = abs(alpha) <= 1e-7
        d_m_small = (1.0 - m) - beta * m
        d_m = d_m_normal * (1 - _mask) + d_m_small * _mask

        alpha = 0.07 * np.exp(-(V + 65.0) / 20.0)
        beta = 1.0 / (np.exp(-(V + 35.0) / 10.0) + 1.0)
        d_h = alpha * (1 - h) - beta * h

        i_na = self.params["g_Na"] * (m**3) * h * (V - self.params["E_Na"])
        i_k = self.params["g_K"] * (n**4) * (V - self.params["E_K"])
        i_l = self.params["g_L"] * (V - self.params["E_L"])

        d_V = I_ext - i_na - i_k - i_l
        return [d_V, d_n, d_m, d_h]
