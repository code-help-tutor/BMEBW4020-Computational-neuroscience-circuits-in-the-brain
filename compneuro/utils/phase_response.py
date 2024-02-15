WeChat: cstutorcs
QQ: 749389476
Email: tutorcs@163.com
"""Phase Response Utilities

This module provides utilities for
1. Computing the iPhase Response Curve using Malkin's Method
2. Computing the Phase Response Curve using Winfree's Method
3. Simulating the Project-Integrate-Fire neuron.
4. Simulating a given BaseModel with Multiplicative Coupling
"""
import inspect
from warnings import warn
import numpy as np
import typing as tp
from collections import OrderedDict
from tqdm.auto import tqdm
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from ..base_model import BaseModel
from .neuron import limit_cycle
from .signal import spike_detect, spike_detect_local
from .. import errors as err


def _check_is_instance(Model) -> tp.Tuple["BaseModel()", "BaseModel"]:
    """Check if model is subclass or instance of BaseModel

    Returns:
        model: instance of Model
        Model: class definition of Model
    """
    if inspect.isclass(Model):
        if issubclass(Model, BaseModel):
            model = Model()
        else:
            raise TypeError(
                f"Model '{Model.__name__}' is not subclass of "
                "compneuro.base_model.BaseModel."
            )
    else:
        model = Model
        Model = model.__class__
    return model, Model


def iPRC(
    model: BaseModel,
    dt: float,
    verbose: bool = False,
    spike_threshold: float = None,
    N_skip_cycles: int = 3,
    spike_variable: str = "V",
    normalize: bool = False,
    normalization_bias: float = 1e-15,
    lc: np.ndarray = None,
    limit_cycle_kws: tp.Mapping[str, float] = None,
    **injected_stimuli,
) -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the phase response curve for a specified inject current value

    Arguments:
        model: instance of an oscillator class inherited from the parent class :code:`BaseModel`
        dt: temporal resolution for the ODE updates (in seconds)
        verbose: whether to show progress
        spike_threshold: minimum height to define the peak of a spike

            .. deprecated:: 0.1.3
                Use :code:`**limit_cycle_kws` instead

        N_skip_cycles: number of limit cycles to skip for psi computation to settle
        spike_variable: variable corresponding to voltage in the specified :code:`model`, this
          variable is used by :py:func:`limit_cycle` to find the limit cycle of the model.
        normalize: if `True`, will scale psi by product with gradient at each point along the limit
          cycle. Since the scaling involves a division, a small bias is added to the denominator which
          could result in spikes in the normalized psi. The bias can be controled by the
          `normalization_bias` parameter.
        normalization_bias: the bias term in the normalization of psi, see above.
        lc: limit cycle for the given model. If specified, the limit cycle computation is skipped.
        limit_cycle_kws: arguments to be passed into `limit_cycle` function

            .. seealso:: :py:func:"compneuro.utils.neuron.limit_cycle"

    Keyword Arguments:
        injected_stimuli: key,value pair of stimuls and constant value for that variable

    .. note::

        The model must have an ode method.
        Furthermore, the state variables of the model should be ordered such that
        voltage is the first variable.

    Returns:
        1. period: 1D array of time vector corresponding to the limit cycle
        2. limit_cycle: 2d numpy array of state variables over time
        3. psi: the desired iPRC of shape
          :code:`(number_of_states, period_length)`
    """
    model, Model = _check_is_instance(model)
    limit_cycle_kws = limit_cycle_kws if limit_cycle_kws is not None else {}

    if not hasattr(model, "ode"):
        raise err.CompNeuroUtilsError(
            "Error: this neuron model does not have an ode method"
        )
    if spike_threshold is not None:
        warn(
            (
                "spike_threshold argument is deprecated. Pass additional arguments to "
                "limit_cycle_kws as keyword arguments directly. This warning will become an "
                "error in the next minor release."
            ),
            DeprecationWarning,
        )
        if "height" not in limit_cycle_kws:
            if np.isscalar(spike_threshold):
                limit_cycle_kws.update({"height": spike_threshold})

    if lc is not None:
        assert lc.ndim == 2 and lc.shape[0] == len(
            model.Default_States
        ), f"Provided limit cycle is of shape ({lc.shape}), mismatched with model."
        t_lc = np.arange(lc.shape[1]) * dt
    else:
        spike_detect_args = {
            var: limit_cycle_kws.pop(var)
            for var in set(limit_cycle_kws.keys())
            - set(inspect.getfullargspec(limit_cycle).args)
        }
        t_lc, lc = limit_cycle(
            model,
            dt=dt,
            verbose=verbose,
            spike_variable=spike_variable,
            spike_detect_args=spike_detect_args,
            **limit_cycle_kws,
            **injected_stimuli,
        )
    period_lc = t_lc.max()

    # NOTE: for autonomous system that does not explicitly depend on `t`,
    # the following line could be replaced by
    # >>> np.vstack(model.ode(states=lc, t=0., I_ext=I_bias))
    # However, it is not easy to know if the model ode is autonomous.
    gradients = np.vstack(
        [
            np.array(model.ode(states=_lc, t=_t, **injected_stimuli)).T
            for _t, _lc in zip(t_lc, lc.T)
        ]
    ).T

    # Step 1: estimate jacobians
    jacc = []
    if model.jacobian is None:  # no jacobian, has to estimate numerically
        dx = 1e-5
        dX = dx * np.eye(len(model.states))  # perturb each state independently
        for tt, _t in enumerate(t_lc):
            A = (
                np.array(model.ode(states=(lc[:, tt] + dX).T, t=_t, **injected_stimuli))
                - np.squeeze(model.ode(states=lc[:, tt], t=_t, **injected_stimuli))[
                    :, None
                ]
            ) / dx
            jacc.append(A)
    else:
        for tt, (_t, _states) in enumerate(zip(t_lc, lc.T)):
            jacc.append(model.jacobian(_t, _states, **injected_stimuli))
    jacc = np.stack(jacc)

    # Step 2: find fundamental set of solutions by solving matrix DE
    #   described by set of equations dF/dt = J(t, x)^T * F, F(0) = I
    #   where J(t,x) is the jacobian of the system at time t with state vector x.
    fund = np.zeros((len(t_lc), len(model.states), len(model.states)))
    F = np.eye(len(model.states))
    for tt, _t in enumerate(t_lc):
        d_F = jacc[tt].T @ F
        F += dt * model.Time_Scale * d_F
        fund[tt] = F

    # Step 3: find the eigen vector with largest eigenvalue
    l, phi = np.linalg.eig(fund[-1].T)
    idx = np.argsort(np.abs(l))
    phi_1 = np.abs(phi[:, idx[-1]])

    # normalize to enforce constant gradient of  phi.T @ f(x^0)
    phi_1 /= phi_1.T @ np.squeeze(model.ode(states=lc[:, 0], t=0, **injected_stimuli))

    # Step 4: find psi by back integrating the adjoint equation
    psi = np.zeros((len(t_lc), len(model.states)))
    psi[0] = phi_1
    for _ in np.arange(N_skip_cycles + 1):
        # enforce boundary condition with normalization
        psi[-1] = psi[0] / (
            psi[0].T @ np.squeeze(model.ode(states=lc[:, 0], t=0, **injected_stimuli))
        )
        for tt in np.arange(len(t_lc) - 1, 0, -1):
            d_PRC = jacc[tt].T @ psi[tt]
            psi[tt - 1] = psi[tt] + dt * model.Time_Scale * d_PRC

    psi = psi.T  # reshape to (N_states, N_t)
    if normalize:
        psi /= normalization_bias + np.sum((psi * gradients), axis=0)[None, :]
    return t_lc, lc, psi


def PRC(
    model: BaseModel,
    dt: float,
    verbose: bool = False,
    spike_threshold: float = 0,
    N_skip_cycles: int = 3,
    spike_variable: str = "V",
    winfree_perturb_amp: tp.Union[float, np.ndarray] = 0.1,
    winfree_steps: int = None,
    normalize: bool = False,
    normalization_bias: float = 1e-15,
    solver: str = None,
    lc: np.ndarray = None,
    limit_cycle_kws: tp.Mapping[str, tp.Any] = None,
    **injected_stimuli,
) -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the phase response curve for a specified inject current value

    Arguments:
        model: instance of an oscillator class inherited from the parent class :code:`BaseModel`
        dt: temporal resolution for the ODE updates (in seconds)
        verbose: whether to show progress
        spike_threshold: minimum height to define the peak of a spike

            .. deprecated:: 0.1.3
                Use :code:`**limit_cycle_kws` instead

        N_skip_cycles: number of limit cycles to skip for PRC computation to settle
        spike_variable: variable corresponding to voltage in the specified :code:`model`, this
          variable is used by :py:func:`limit_cycle` to find the limit cycle of the model.
        winfree_perturb_amp: perturbation amplitude of the winfree's method. If scalar, the same
          amplitude is used for all state variables.

        .. note::

            The unit of :code:`winfree_perturb_amp` is actually (variable_unit / model_time_unit).
            In the case of voltage for Hodging-Huxley neuron, this is mili-Volt/mili-Second

        winfree_steps: number of steps to compute winfree's PRC, defaults to entire limit cycle at
          :code:`dt` resolution
        normalize: if `True`, will scale PRC by product with gradient at each point along the limit
          cycle. Since the scaling involves a division, a small bias is added to the denominator which
          could result in spikes in the normalized PRC. The bias can be controled by the
          `normalization_bias` parameter.
        normalization_bias: the bias term in the normalization of PRC, see above.
        solver: solver to use for ivp
        lc: limit cycle for the given model. If specified, the limit cycle computation is skipped.
        limit_cycle_kws: arguments to be passed into `limit_cycle` function

            .. seealso:: :py:func:"compneuro.utils.neuron.limit_cycle"

    Keyword Arguments:
        injected_stimuli: key,value pair of stimuls and constant value for that variable

    .. note::

        The model must have an ode method.
        Furthermore, the state variables of the model should be ordered such that voltage is the
        first variable.

    Returns:
        1. period: 1D array of time vector corresponding to the limit cycle
        2. limit_cycle: 2d numpy array of state variables over time
        3. prc: the desired phase response curves of shape
          :code:`(number_of_states, period_length)`
    """
    model, Model = _check_is_instance(model)

    limit_cycle_kws = {} if limit_cycle_kws is None else limit_cycle_kws
    if not hasattr(model, "ode"):
        raise err.CompNeuroUtilsError(
            "Error: this neuron model does not have an ode method"
        )

    if spike_threshold is not None:
        warn(
            (
                "spike_threshold argument is deprecated. Pass additional arguments to "
                "limit_cycle_kws as keyword arguments directly. This warning will become an "
                "error in the next minor release."
            ),
            DeprecationWarning,
        )
        if "height" not in limit_cycle_kws:
            if np.isscalar(spike_threshold):
                limit_cycle_kws.update({"height": spike_threshold})

    if lc is not None:
        assert lc.ndim == 2 and lc.shape[0] == len(
            model.Default_States
        ), f"Provided limit cycle is of shape ({lc.shape}), mismatched with model."
        t_lc = np.arange(lc.shape[1]) * dt
    else:
        spike_detect_args = {
            var: limit_cycle_kws.pop(var)
            for var in set(limit_cycle_kws.keys())
            - set(inspect.getfullargspec(limit_cycle).args)
        }
        t_lc, lc = limit_cycle(
            model,
            dt=dt,
            verbose=verbose,
            spike_variable=spike_variable,
            spike_detect_args=spike_detect_args,
            **limit_cycle_kws,
            **injected_stimuli,
        )
    period_lc = t_lc.max()

    # NOTE: for autonomous system that does not explicitly depend on `t`,
    # the following line could be replaced by
    # >>> np.vstack(model.ode(states=lc, t=0., **injected_stimuli))
    # However, it is not easy to know if the model ode is autonomous.
    gradients = np.vstack(
        [
            np.array(model.ode(states=_lc, t=_t, **injected_stimuli)).T
            for _t, _lc in zip(t_lc, lc.T)
        ]
    ).T

    prc = np.zeros((len(model.Default_States), len(t_lc)))

    ### BEGIN SOLUTION (tag:2)
    if winfree_steps is None:
        t_idx_perturbed = np.arange(len(t_lc))
        winfree_steps = len(t_idx_perturbed)
    else:
        t_idx_perturbed = np.linspace(0, len(t_lc) - 1, winfree_steps).astype(int)

    if np.isscalar(winfree_perturb_amp):
        winfree_perturb_amp = np.full(len(model.states), winfree_perturb_amp)
    else:
        winfree_perturb_amp = np.atleast_1d(winfree_perturb_amp)
        if len(winfree_perturb_amp) != len(model.states):
            raise Exception(
                f"Model {model.__class__.__name__} has {len(model.states)} states, but "
                f"perturbation amplitude is length {len(winfree_perturb_amp)}."
            )

    # create initial condition for perturbed neuron model for each state
    # variable at each one of the `winfree_steps` along the limit cycle.
    perturbation = np.diag(winfree_perturb_amp)
    x0_perturbed = np.vstack(
        [
            lc[:, t_idx_perturbed].T + p[None, :]
            for p in perturbation  # perturbe each state independently
        ]
    )
    x0_perturbed = x0_perturbed.T  # shape (N_states, N_states * N_t)

    # create batched model where each neuron has state initialized as one of
    # the perturbed state
    model_perturbed = model.__class__(
        num=winfree_steps * len(model.states),
        **{_key: _x0 for _key, _x0 in zip(model.states.keys(), x0_perturbed)},
    )

    # simulate perturbed neuron for at least N_skip_cycles + 3 cycles. This
    # is to ensure that we get at least N_skip_cycles spikes out of the
    # perturbed orbits.
    t_perturbed_solve = np.arange(len(t_lc) * (N_skip_cycles + 3)) * dt
    res_perturbed = model_perturbed.solve(
        t_perturbed_solve,
        **{
            var_name: np.full(t_perturbed_solve.shape, val)
            for var_name, val in injected_stimuli.items()
        },
        solver=solver,
        verbose=f"PRC (Winfree) - {model.__class__.__name__}" if verbose else False,
    )

    # find the first N_skip_cycles spikes, the N_skip_cycles+1 spike of the
    # unperturbed orbit is compared against the new perturbed spike time.
    prc_downsample = np.zeros(len(t_lc[t_idx_perturbed]) * len(model.states))

    # find entries in limit_cycle_kws that can be used for spike_detect
    spike_detect_args = {
        key: val
        for key, val in limit_cycle_kws.items()
        if key in find_peaks.__code__.co_varnames
    }
    spike_mask = spike_detect(res_perturbed[spike_variable], **spike_detect_args)
    for n, ss in enumerate(spike_mask):
        (tk_idx,) = np.where(ss)
        if len(tk_idx) <= N_skip_cycles:
            raise err.CompNeuroUtilsError(
                f"Only {len(tk_idx)} spikes found after perturbation, "
                f"desired {N_skip_cycles}."
            )
        T_old = period_lc * (N_skip_cycles + 1)
        t_start = t_lc[t_idx_perturbed[n % winfree_steps]]
        T_new = t_start + t_perturbed_solve[tk_idx[N_skip_cycles]]

        # The PRC is defined as (T_old - T_new) (in unit of `dt`)
        # NOTE: see Izhikevich page 455 for PRC winfree definition
        prc_downsample[n] = T_old - T_new

    # Reshape PRC into shape (len(states), N_t)
    prc_downsample = prc_downsample.reshape((len(model.states), -1))

    # For ease of computation, we first normalize the PRC by period T to
    # Normalize to range [-1,1]. We will eventually put it back to the
    # original unit.
    prc_downsample /= period_lc

    # Wrap normalized PRC to range [-1/2, 1/2]. For example, if a PRC value
    # is 1, this means that the new period is off by an entire period, which
    # is definitely an artifact of spike detector and we just set this value
    # to 0.
    prc_downsample = (prc_downsample + 0.5) % 1.0 - 0.5

    # Normalize PRC to have unit of (model_time_unit / state_variable_unit)
    # It is done this way so that the unit works out if the Phase-Shift-Process
    # definition, where the product of the PRC and the gradient of the ODE
    # is unit-less.
    prc_downsample *= period_lc  # in unit of dt (usually second)
    prc_downsample *= model.Time_Scale  # in model time_unit
    prc_downsample /= winfree_perturb_amp[
        :, None
    ]  # in model_time_unit/state_variable_unit # * dt * model.Time_Scale

    # Finally, interpolate the downsampled PRC to the entire limit cycle
    prc = interp1d(t_lc[t_idx_perturbed], prc_downsample, axis=-1)(t_lc)

    # Enforce boundary condition
    prc[:, 0] = prc[:, -1]
    if normalize:
        prc /= normalization_bias + np.sum((prc * gradients), axis=0)[None, :]

    ### END SOLUTION

    return t_lc, lc, prc


def solve_multiplicative(
    model, t, stimulus: np.ndarray, verbose=False, **injected_stimuli
):
    """Solve model equation under multiplicative coupling

    Arguments:
        model: instance of a class inherited from the parent class :code:`BaseModel`
        t: 1d numpy array of time vector of the simulation
        stimulus: time-dependent signal that drive the neuron multiplicatively
        verbose: whether to show progress

    Keyword Arguments:
        injected_stimuli: key value pairt of injected stimuli that make the neuron
          oscillate periodically

    .. note::

        The model must have an ode method.

    Returns:
        A dictionary of simulation results keyed by state variables and
        each entry is of shape :code:`(len(t),)`
    """
    model, Model = _check_is_instance(model)

    if not hasattr(model, "ode"):
        raise err.CompNeuroUtilsError(
            "Error: this neuron model does not have an ode method"
        )

    # Scale the time according to the intrinsic model timescale
    t_long = t * model.Time_Scale
    d_t = t_long[1] - t_long[0]

    res = np.zeros((len(t_long), len(model.state_arr), 1))
    model.reset()
    res[0] = model.state_arr

    iters = enumerate(zip(stimulus[:-1], t_long[:-1]))
    if verbose:
        iters = tqdm(iters, total=len(t_long) - 1)

    for tt, (_stim, _t) in iters:
        d_x = _stim * np.vstack(
            model.ode(states=model.state_arr, **injected_stimuli, t=_t)
        )
        model.state_arr += d_t * d_x
        model.clip()
        res[tt + 1] = model.state_arr

    res = np.moveaxis(res, 0, 2)
    res = OrderedDict({key: res[n] for n, key in enumerate(model.states.keys())})
    return res


def PIF(t, u, prc_V, V0=0, verbose=False):
    """Compute the voltage trace output from the equivalent project-integrate-fire (PIF) neuron

    Arguments:
        t: 1d numpy array of time vector
        u: 1d numpy array of current input
        prc_V: 1d numpy array of the phase response curve of the reference neuron
          for voltage
        V0: initial condition for voltage

    Returns:
        V: 1d numpy array of simulated output voltage trace of the PIF neuron
    """
    dt = t[1] - t[0]
    if prc_V.ndim > 1:
        raise err.CompNeuroUtilsError(
            "PRC for Voltage should be 1D array, got array with shape "
            f"{prc_V.shape} instead."
        )
    T = len(prc_V) * dt
    V = np.zeros_like(t)

    iters = enumerate(t)
    if verbose:
        iters = tqdm(iters, total=len(t), desc="PIF Neuron")

    prc_ctr = 0
    for tt, _t in iters:
        if tt == 0:
            V_old = V0
        else:
            V_old = V[tt - 1]

        d_V = 1 + prc_V[prc_ctr] * u[tt]
        V_new = V_old + dt * d_V

        prc_ctr = (prc_ctr + 1) % len(prc_V)
        if V_new >= T:
            V_new = 0
            prc_ctr = 0
        V[tt] = V_new
    return V
