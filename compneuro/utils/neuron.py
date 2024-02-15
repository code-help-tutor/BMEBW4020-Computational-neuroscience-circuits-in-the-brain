WeChat: cstutorcs
QQ: 749389476
Email: tutorcs@163.com
"""Define Utility Functions for Neuorns
"""
import numpy as np
import typing as tp
import inspect
from warnings import warn
from tqdm.auto import tqdm
from .. import errors as err
from .signal import spike_detect, spike_detect_local
from ..base_model import BaseModel


def limit_cycle(
    model: BaseModel,
    dt: float = 1e-5,
    N_spikes: int = 5,
    max_dur: float = 1.0,
    spike_variable: str = "V",
    spike_threshold: float = None,
    verbose: bool = True,
    solver: str = "Euler",
    spike_detect_args: tp.Mapping[str, tp.Any] = None,
    **injected_stimuli,
) -> tp.Tuple[np.ndarray, np.ndarray]:
    """Compute Limit Cycle of Neuron Model

    Arguments:
        Model: Neuron Model to compute Limit Cycle of
        dt: time resolution
        N_spikes: number of spikes to detect before terminating limit cycle
          computation
        max_dur: maximum duration of simulation before terminating in seconds
        spike_variable: string name of the variable corresponding to voltage
        spike_threshold: height of the spike variable state value

            .. deprecated:: 0.1.3
                Use :code:`**spike_detect_args` instead

        verbose: whether to show progress bar
        solver: which solver to use. Note that if any solver other than "Euler"
          is specified, the entire result of duration `max_dur` will be simulated
          since callbacks are only supported in the Euler's method at the
          moment.

            .. seealso:: :py:func:"compneuro.base_model.BaseModel.solve"

        spike_detect_args: arguments to be passed into `spike_detect` function

            .. seealso:: :py:func:"compneuro.utils.signal.spike_detect"

    Keyword Arguments:
        injected_stimuli: key,val pair of stimuli to be provided at constant value
          for the model to obtain limit cycle. The variable names must match
          those used by the neuron model's :code:`ode` method.

    Returns:
        A tuple of `(t_lc, lc)` where:

          1. `t_lc` is a 1D array of time vector corresponding to the limit cycle in
            the unit of argument :code:`dt`.
          2. `lc` is a 2D array of shape :code:`(N_states, len(t_lc))` where the
            states are listed in the same order as the model's :code:`.states` attr

    Raises:
        compneuro.errors.CompNeuroUtilsError: raised if

          1. `model` is not the right type
          2. no limit cycle is found
    """
    spike_detect_args = {} if spike_detect_args is None else spike_detect_args

    _missing_inputs = set(model._input_args) - set(injected_stimuli.keys())
    if _missing_inputs:
        warn(
            (
                f"Stimuli'{_missing_inputs}' not specified as limit_cycle arguments, "
                "this is likely not the intended behavior."
            ),
            err.CompNeuroUtilsWarning,
        )

    _extraneous_inputs = set(injected_stimuli.keys()) - set(model._input_args)
    if _extraneous_inputs:
        raise err.CompNeuroUtilsError(
            f"Stimulus '{_extraneous_inputs}' not found in {model.__class__.__name__}'s inputs, "
            f"must be one of '{model._input_args}'"
        )

    if inspect.isclass(model):
        neu = model()
    else:
        neu = model
    if spike_variable not in model.states:
        raise err.CompNeuroUtilsError(
            f"Spike Variable '{spike_variable}' not found in model class "
            f"{model.__class__.__name__}. Must be one of {list(model.states.keys())}"
        )
    if spike_threshold is not None:
        warn(
            (
                "spike_threshold argument is deprecated. Pass additional arguments to "
                "spike_detect as keyword arguments directly. This warning will become an "
                "error in the next minor release."
            ),
            DeprecationWarning,
        )
        if "height" not in spike_detect_args:
            if np.isscalar(spike_threshold):
                spike_detect_args.update({"height": spike_threshold})

    t = np.arange(0, max_dur, dt)
    res_states = np.zeros((len(t), len(model.states), 1))

    if solver == "Euler":
        t_idx = 0
        tk_last = -np.inf
        V_state_index = list(model.states.keys()).index(spike_variable)
        V_prev2 = np.inf
        V_prev = np.inf
        V_curr = np.inf
        spike_count = 0
        distance = (
            spike_detect_args["distance"] if "distance" in spike_detect_args else 0
        )
        height = spike_detect_args["height"] if "height" in spike_detect_args else 0.0

        def spike_callback(neuron):
            nonlocal V_prev2, V_prev, V_curr, spike_count, t_idx, tk_last, res_states
            V_prev2 = V_prev
            V_prev = V_curr
            V_curr = neuron.state_arr[V_state_index]
            res_states[t_idx] = neuron.state_arr
            _spike = int(
                np.logical_and(
                    spike_detect_local(V_prev2, V_prev, V_curr, height),
                    t_idx - tk_last >= distance,
                )
            )
            if _spike:
                tk_last = t_idx
            spike_count += _spike
            t_idx += 1
            if spike_count >= N_spikes:
                raise StopIteration

        try:
            _ = neu.solve(
                t,
                verbose=verbose,
                callbacks=spike_callback,
                solver="Euler",
                **{
                    var_name: np.full(t.shape, stim)
                    for var_name, stim in injected_stimuli.items()
                },
            )
        except StopIteration:
            pass
        else:  # no exception, means not enough spikes found
            raise err.CompNeuroUtilsError(
                "Insufficient spikes found during max duration. Asked for "
                f"{N_spikes} got {spike_count}."
            )
        res_V = res_states[:, V_state_index, 0]
        # Extract Limit Cycle from the consant injected stimuli
        (tk_idx,) = np.where(spike_detect(res_V[:t_idx], **spike_detect_args))
    else:
        res = neu.solve(
            t,
            verbose=verbose,
            solver=solver,
            **{
                var_name: np.full(t.shape, stim)
                for var_name, stim in injected_stimuli.items()
            },
        )
        res_V = res[spike_variable][0]
        (tk_idx,) = np.where(spike_detect(res_V, **spike_detect_args))
        if len(tk_idx) < N_spikes:
            raise err.CompNeuroUtilsError(
                "Insufficient spikes found during max duration. Asked for "
                f"{N_spikes} got {len(tk_idx)}."
            )
        res_states = np.stack([v.T for v in res.values()], axis=1)

    lc_slice = slice(tk_idx[-2], tk_idx[-1])
    lc = res_states[lc_slice, :, 0]
    t_lc = t[lc_slice]
    t_lc -= t_lc[0]

    return t_lc, lc.T