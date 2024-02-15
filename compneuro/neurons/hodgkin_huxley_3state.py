WeChat: cstutorcs
QQ: 749389476
Email: tutorcs@163.com
import numpy as np
from collections import OrderedDict
from compneuro.base_model import BaseModel


class HodgkinHuxley3State(BaseModel):
    """Hodgkin Huxley Neuron Model with 3 States"""

    Time_Scale = 1e3  # s to ms
    """Hodgkin-Huxley Reduced Model Operates on Milisecond Scale"""
    Default_States = OrderedDict(V=-60, n=(0.0, 0.0, 1.0), h=(1.0, 0.0, 1.0))
    """Default State Variables of the Hodgkin-Huxley Reduced Model"""
    Default_Params = OrderedDict(
        g_Na=120.0, g_K=36.0, g_L=0.3, E_Na=50.0, E_K=-77.0, E_L=-54.387
    )
    """Default Parameters of the Hodgkin-Huxley Reduced Model"""

    def ode(self, t, states, I_ext=0.0):
        """Hodgkin-Huxley gradient function"""
        V, n, h = states

        alpha = np.exp(-(V + 55.0) / 10.0) - 1.0
        beta = 0.125 * np.exp(-(V + 65.0) / 80.0)
        d_n_normal = (-0.01 * (V + 55.0) / alpha) * (1.0 - n) - beta * n
        d_n_small = 0.1 * (1.0 - n) - beta * n
        _mask = abs(alpha) <= 1e-7
        d_n = d_n_normal * (1 - _mask) + d_n_small * _mask

        alpha = 0.1 * (25 - (V + 65)) / (np.exp((25 - (V + 65)) / 10) - 1)
        beta = 4 * np.exp(-(V + 65) / 18)
        m_infty = alpha / (alpha + beta)

        alpha = 0.07 * np.exp(-(V + 65.0) / 20.0)
        beta = 1.0 / (np.exp(-(V + 35.0) / 10.0) + 1.0)
        d_h = alpha * (1 - h) - beta * h

        i_na = self.params["g_Na"] * (m_infty**3) * h * (V - self.params["E_Na"])
        i_k = self.params["g_K"] * (n**4) * (V - self.params["E_K"])
        i_l = self.params["g_L"] * (V - self.params["E_L"])

        d_V = I_ext - i_na - i_k - i_l
        return [d_V, d_n, d_h]
