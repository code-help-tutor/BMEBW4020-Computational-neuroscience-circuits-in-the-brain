WeChat: cstutorcs
QQ: 749389476
Email: tutorcs@163.com
# pylint:disable=C0103
"""Base Class for ODE-based Models
"""
import sys
import typing as tp
import inspect
from warnings import warn
from collections import OrderedDict
import numpy as np
from tqdm.auto import tqdm  # pylint:disable
from scipy.integrate import solve_ivp, odeint
from scipy.integrate._ivp.base import OdeSolver
from scipy.interpolate import interp1d

from . import errors as err
from . import types as tpe
from .model_parser import ParsedModel
from .utils.model_viewer import ModelViewer

PY37 = sys.version_info.major * 10 + sys.version_info.minor >= 37

IVP_SOLVER_WITH_JACC = ("radau", "bdf", "lsoda")


class BaseModel:
    """Base Model Class

    This base class is an opinionated implementation of ODE-based models
    for Comp Neuro.

    Arguments:
        num: number of state variables to simulate simultaneously
        kwargs: keyword arguments that overwrite initial conditions of state
            variables and values of parameters Can either be a `scalar` or
            a 1D numpy array with the dimensionality of :code:`(num,)`
    """

    Time_Scale: float = 1.0
    """Scaling for the time-step :code:`dt`. For models that must be simulated
    on a milisecond scale, set to :code:`1000`."""

    Default_Params: OrderedDict = OrderedDict()
    """A dictionary of (name, value) pair for parameters"""

    Default_States: OrderedDict = OrderedDict()
    """A dictionary of state variables where each entry is either

    - scalar: initial value of state variable
    - 3 tuple: (initial value, min, max) of the state variable

    .. warning::

        Due to a lack of variable bounding API in scipy, bounding variables
        are currently only available for :code:`Euler` method when solving
        ODEs.

        .. seealso:: :py:func:`solve`
    """

    Supported_Solvers: tp.Tuple[tp.Union[str, tp.Callable]] = (
        "RK45",
        "Euler",
        "odeint",
        "LSODA",
        "RK23",
        "DOP853",
        "Radau",
        "BDF",
    )
    """A tuple of all supported solvers for the model"""

    callbacks: tp.Tuple[tp.Callable] = ()
    """A tuple of all callback functions after each solve step

    .. note::

        Only applicable when solver is `Euler`
    """

    def __init__(self, num: int = 1, callbacks=None, **kwargs):

        # check ordering of the states
        if not PY37 and not isinstance(self.Default_States, OrderedDict):
            warn(
                "States need to be OrderedDict to maintain variable ordering, "
                "converting...",
                err.CompNeuroWarning,
            )
            self.Default_States = OrderedDict(self.Default_States)

        # check to make sure that the states and parameters don't have overlapping
        # names
        duplicate_vars = set(self.Default_Params.keys()).intersection(
            set(self.Default_States.keys())
        )
        if len(duplicate_vars) > 0:
            raise err.CompNeuroModelError(
                f"Params and States cannot duplicate names: {duplicate_vars}"
            )

        # parse the input arguments of the model
        _argspecs = inspect.getfullargspec(self.ode)
        self._input_args = tuple(
            [
                var
                for var in _argspecs.args + _argspecs.kwonlyargs
                if var not in ["self", "t", "states"]
            ]
        )

        self.num = num  # number of neurons instantiated
        self.params = OrderedDict(self.Default_Params.copy())
        self.states = OrderedDict()
        self.initial_states = OrderedDict()
        self.bounds = OrderedDict()

        for var_name, var_val in self.Default_States.items():
            var_val = np.atleast_1d(var_val)
            if len(var_val) == 1:
                self.initial_states[var_name] = var_val[[0]].copy()
                self.states[var_name] = var_val[[0]].copy()
                self.bounds[var_name] = None
            elif len(var_val) == 3:
                self.initial_states[var_name] = var_val[[0]].copy()
                self.states[var_name] = var_val[[0]].copy()
                self.bounds[var_name] = var_val[1:].copy()
            else:
                raise err.CompNeuroModelError(
                    f"Expect state variable {var_name} to have length 1 or 3, "
                    f"got {len(var_val)}"
                )

        for key, val in kwargs.items():
            if key in self.params:
                self.params[key] = val
            elif key in self.states:
                self.states[key] = val
                self.initial_states[key] = val
            else:
                raise err.CompNeuroModelError(f"Unrecognized argument {key}")
        self.__check_dimensions()

        if callbacks is not None:
            callbacks = tuple(np.atleast_1d(callbacks).tolist())
            for f in callbacks:
                if not callable(f):
                    raise err.CompNeuroModelError("Callback is not callable\n" f"{f}")
            self.callbacks = callbacks

        # Parsed Model and Jacobian will be computed the first time that
        # `self.jacobian` property is invoked
        self._parsed_model = None
        self._jacobian = None
        self.compute_jacobian()

        # pretty printed model viewer, intialized on first call of _repr_html_
        self._model_viewer = None

    def _ipython_display_(self):
        """Pretty print model if possible"""
        from IPython.display import display  # pylint:disable=import-outside-toplevel

        if self._model_viewer is None:
            self._model_viewer = ModelViewer(self)
        display(self._model_viewer.tab)

    def compute_jacobian(self) -> tp.Callable:
        """Compute Jacobian of Model

        .. note::

            Differing from :py:func:`jacobian`, this function will always
            `re-compute` jacobian, including re-parsing the model. This is
            provided in case the model parameter has been changed in-place
            and the jacobian needs to be updated.

        Returns:
            A callable :code:`jacc_f(t, states, I_ext)` that returns a 2D numpy
            array corresponding to the jacobian of the model. For model that does
            not require `I_ext` input, the callable's call signature is
            :code:`jacc_f(t, states, I_ext)`.
        """
        jacc_f = None

        try:
            from sympy import lambdify

            self._parsed_model = ParsedModel(self)
            jacc = self._parsed_model.jacobian()
            arguments = [
                val
                if name != "states"
                else tuple(self._parsed_model.state_vars.values())
                for name, val in self._parsed_model.input_vars.items()
            ]
            jacc_f = lambdify(arguments, jacc)

        except Exception as e:
            warn(
                (
                    f"Model parser failed for '{self.__class__.__name__}', "
                    "this will disable automatic jacobian support. Traceback:\n"
                    f"{repr(e)}"
                ),
                err.CompNeuroWarning,
            )

        # set the jacobian for model
        self._jacobian = jacc_f
        return jacc_f

    @property
    def jacobian(self) -> tp.Callable:
        """Compute or return cached jacobian of the model

        .. note::

            You can override jacobian definition in child classes to enforce
            a jacobian

        .. seealso:: :py:func:`compNeuro.BaseModel.compute_jacobian`

        Returns:
            A callable :code:`jacc_f(t, states, I_ext)` that returns a 2D numpy
            array corresponding to the jacobian of the model. For model that does
            not require `I_ext` input, the callable's call signature is
            :code:`jacc_f(t, states, I_ext)`.
        """
        if self._jacobian is not None:
            return self._jacobian
        return self.compute_jacobian()

    def reset(self) -> None:
        """Reset Initial Values of the System"""
        for var_name, var_val in self.initial_states.items():
            var_val = np.atleast_1d(var_val)
            self.states[var_name] = var_val.copy()

    @property
    def state_arr(self) -> np.ndarray:
        """State Vector for Batched ODE Solver

        This attribute stackes all state values into a
        :code:`(len(self.states), self.num)` shaped array. It is done for ODE
        solver to handle the state variable easily.
        """
        return np.vstack(list(self.states.values()))

    @state_arr.setter
    def state_arr(self, new_value) -> None:
        """Settting state_vector set states dictionary

        The setter and getter for state_arr is intended to ensure consistency
        between `self.states` and `self.state_arr`
        """
        for var_name, new_val in zip(self.states.keys(), new_value):
            self.states[var_name] = new_val

    def __check_dimensions(self):
        """Ensure consistent dimensions for all parameters and states"""

        # make sure vector-valued parameters have the same shape as the number
        # of components in the model
        for key, val in self.params.items():
            if np.isscalar(val):
                continue
            else:
                if len(val) == 1:
                    self.params[key] = np.repeat(val, self.num)
                else:
                    if len(val) != self.num:
                        raise err.CompNeuroModelError(
                            f"Parameter '{key}'' should have scalar value or array of "
                            f"length num={self.num}, got {len(val)} instead."
                        )

        # Ensure states have the same shape as number of components in the model
        # Convert all scalar-valued states to 1D vector of length `(self.num,)`
        for key, val in self.states.items():
            if np.isscalar(val):
                self.states[key] = np.full((self.num,), val)
            else:
                if len(val) == 1:
                    self.states[key] = np.repeat(val, self.num)
                else:
                    if len(val) != self.num:
                        raise err.CompNeuroModelError(
                            f"State '{key}' should have scalar value or array of "
                            f"length num={self.num}, got {len(val)} instead."
                        )

        # Ensure initial states have the same shape as states. This will be
        # necessary to ensure `self.reset` changes states to the right shape.
        for key, val in self.initial_states.items():
            if np.isscalar(val):
                self.initial_states[key] = np.full((self.num,), val)
            else:
                if len(val) == 1:
                    self.initial_states[key] = np.repeat(val, self.num)
                else:
                    if len(val) != self.num:
                        raise err.CompNeuroModelError(
                            f"Initial State '{key}' should have scalar value or "
                            f"array of length num={self.num}, got {len(val)} "
                            "instead."
                        )

    def ode(
        self, t: float, states: np.ndarray, stimuli: tp.Union[float, np.ndarray] = None
    ) -> tp.Iterable:
        """Definition of Differential Equation

        .. note::

            For dynamical models described by ODEs, the function
            definition of the model should be of the form
            :math:`dx/dt=f(t,\\mathbf{x})`,
            where :math:`\\mathbf{x}` is usually a vector variable that describes
            the current state values at which the gradient is evaluted, and
            :math:`t` is the current time. Also note that :code:`t` is required
            to be compatible with scipy ode solver's API, although it is not
            needed for `autonomous` ODEs.

        .. seealso:: :py:mod:`scipy.integrate.solve_ivp`
            and :py:mod:`scipy.integrate.odeint`

        Arguments:
            t: current time value
            states: vector of state variable

        Keyword Arguments:
            stimuli: input stimuli, can be multiple.

                .. note:: The name that is given to these stimuli is important since
                  the exact name will be used to match the `**stimuli` keyword arguments
                  in :py:func:`solve`.

        Returns:
            An  vector of gradient value evaluated based on the
            model dynamics. Should have the same shape as the
            input :code:`states` vector.
        """
        raise NotImplementedError

    def solve(
        self,
        t: np.ndarray,
        *,
        I_ext: np.ndarray = None,
        solver: tpe.solvers = None,
        reset_initial_state: bool = True,
        verbose: tp.Union[bool, str] = True,
        full_output: bool = False,
        callbacks: tp.Union[tp.Callable, tp.Iterable[tp.Callable]] = None,
        solver_kws: tp.Mapping[str, tp.Any] = None,
        **stimuli,
    ) -> tp.Union[tp.Dict, tp.Tuple[tp.Dict, tp.Any]]:
        """Solve model equation for entire input

        Positional Arguments:
            t: 1d numpy array of time vector of the simulation

        Keyword-Only Arguments:
            I_ext: external current driving the model

                .. deprecated:: 0.1.3
                    Use :code:`stimuli` keyword argument inputs instead.

            solver: Which ODE solver to use, defaults to the first entry in the
              :code:`Supported_Solvers` attribute.

                - `Euler`: Custom forward euler solver, default.
                - `odeint`: Use :py:mod:`scipy.integrate.odeint` which uses LSODA
                - `LSODA`: Use :py:mod:`scipy.integrate.odeint` which uses LSODA
                - `RK45/RK23/DOP853`: Use
                  :py:mod:`scipy.integrate.solve_ivp` with the specified method
                - :py:mod:`scipy.integrate.OdeSolver` instance: Use
                  :py:mod:`scipy.integrate.solve_ivp` with the provided custom solver

            reset_initial_state: whether to reset the initial state value of the
              model to the values in :code:`Default_State`. Default to True.
            verbose: If is not `False` a progress bar will be created. If is `str`,
              the value will be set to the description of the progress bar.
            full_output: whether to return the entire output from scipy's
              ode solvers.
            callbacks: functions of the signature :code:`function(self)` that is
              executed for :code:`solver=Euler` at every step.
            solver_kws: a dictionary containingarguments to be passed into the ode
              solvers if scipy solvers are used.

                .. seealso: :py:mod:`scipy.integrate.solve_ivp` and
                    :py:mod:`scipy.integrate.odeint`

        .. note::

            String names for :code:`solve_ivp` (RK45/RK23/DOP853)
            are case-sensitive but not for any other methods.
            Also note that the solvers can hang if the amplitude scale of
            :code:`I_ext` is too large.


        Keyword Arguments:
            stimuli: Key value pair of input arguments that matches the signature
              of the :func:`ode` function.

        Returns:
            Return dictionary of a 2-tuple depending on argument
            :code:`full_output`:

            - `False`: An dictionary of simulation results keyed by state
              variables and each entry is of shape :code:`(num, len(t))`
            - `True`: A 2-tuple where the first entry is as above, and the
              second entry is the ode result from either
              :py:mod:`scipy.integrate.odeint` or
              :py:mod:`scipy.integrate.solve_ivp`. The second entry will be
              :code:`None` if solver is :code:`Euler`
        """
        # validate solver
        if solver is None:
            solver = self.Supported_Solvers[0]
        if isinstance(solver, OdeSolver):
            pass
        else:
            if solver not in self.Supported_Solvers:
                raise err.CompNeuroModelError(
                    f"Solver '{solver}' not understood, must be one of "
                    f"{self.Supported_Solvers}."
                )
        solver_kws = {} if solver_kws is None else solver_kws

        # I_ext argument is deprecated unless it's part of stimuli
        if I_ext is not None and "I_ext" not in self._input_args:
            raise err.CompNeuroModelError(
                "I_ext is deprecated since 0.1.3. Instead, provide input array as "
                "keyword arguments "
                "where the keyword names match with the call signature of model.ode() "
                "ignoring `t, states`. For this model, the valid keywords for input "
                f"stimuli are {self._input_args}"
            )
        elif I_ext is not None and "I_ext" in self._input_args:
            stimuli.update({"I_ext": I_ext})

        # validate stimuli - check to make sure that the keyword arguments contain only
        # arguments that are relevant to the model input.
        if stimuli:
            _extraneous_input_args = set(stimuli.keys()) - set(self._input_args)
            _missing_input_args = set(self._input_args) - set(stimuli.keys())
            if _extraneous_input_args:
                raise err.CompNeuroModelError(
                    (
                        f"Extraneous input arguments '{_extraneous_input_args}' "
                        "treated as stimuli but are not found in the function "
                        f"definition of {self.__class__.__name__}.ode(), "
                        f"the only supported input variables are '{self._input_args}'"
                    )
                )
            if _missing_input_args:
                raise err.CompNeuroModelError(
                    f"Input argument '{_missing_input_args}' missing but are required "
                    f"by the {self.__class__.__name__}.ode() method. Please provide "
                    f"all inputs in '{self._input_args}'."
                )

        # whether to reset initial state to `self.initial_states`
        if reset_initial_state:
            self.reset()

        # rescale time axis appropriately
        t_long = t * self.Time_Scale
        state_var_shape = self.state_arr.shape
        x0 = np.ravel(self.state_arr)
        d_t = t_long[1] - t_long[0]

        # check external current dimension. It has to be either 1D array the
        # same shape as `t`, or a 2D array of shape `(len(t), self.num)`
        if stimuli:
            for var_name, stim in stimuli.items():
                if stim.ndim == 1:
                    stim = np.repeat(stim[:, None], self.num, axis=-1)
                elif stim.ndim != 2:
                    raise err.CompNeuroModelError(
                        f"Stimulus '{var_name}' must be 1D or 2D"
                    )
                if len(stim) != len(t):
                    raise err.CompNeuroModelError(
                        f"Stimulus '{var_name}' first dimesion must be the same length as t"
                    )
                if stim.shape[1] > 1:
                    if stim.shape != (len(t_long), self.num):
                        raise err.CompNeuroModelError(
                            f"Stimulus '{var_name}' expects shape ({len(t_long)},{self.num}), "
                            f"got {stim.shape}"
                        )
                stimuli[var_name] = stim

        # Register callback that is executed after every euler step.
        callbacks = [] if callbacks is None else np.atleast_1d(callbacks).tolist()
        for f in callbacks:
            if not callable(f):
                raise err.CompNeuroModelError("Callback is not callable\n" f"{f}")
        callbacks = tuple(list(self.callbacks) + callbacks)
        if len(callbacks) > 0 and solver.lower() != "euler":
            warn(
                f"Callback only supported for Euler's method, got '{solver}'",
                err.CompNeuroWarning,
            )
            callbacks = None

        # Solve Euler's method
        if solver.lower() == "euler":  # solver euler
            res = np.zeros((len(t_long), len(self.state_arr), self.num))
            # run loop
            iters = enumerate(t_long)
            if verbose:
                iters = tqdm(
                    iters,
                    total=len(t_long),
                    desc=verbose if isinstance(verbose, str) else "",
                )

            for tt, _t in iters:
                _stim = {var_name: stim[tt] for var_name, stim in stimuli.items()}
                d_x = np.vstack(self.ode(_t, self.state_arr, **_stim))
                self.state_arr += d_t * d_x
                self.clip()
                res[tt] = self.state_arr
                if callbacks is not None:
                    for _func in callbacks:
                        _func(self)
            # move time axis to last so that we end up with shape
            # (len(self.states), self.num, len(t))
            res = np.moveaxis(res, 0, 2)
            res = OrderedDict({key: res[n] for n, key in enumerate(self.states.keys())})

            if full_output:
                return res, None
            return res

        # Solve IVP Methods
        if verbose:
            pbar = tqdm(
                total=len(t_long), desc=verbose if isinstance(verbose, str) else ""
            )

        # 1. create update function for IVP
        jacc_f = None
        if stimuli:  # has external input
            interpolators = {
                var_name: interp1d(
                    t_long, stim, axis=0, kind="zero", fill_value="extrapolate"
                )
                for var_name, stim in stimuli.items()
            }
            if self.jacobian is not None:
                # rewrite jacobian function to include invaluation at input value
                def jacc_f(t, states):  # pylint:disable=function-redefined
                    return self.jacobian(  # pylint:disable=not-callable
                        t,
                        states,
                        **{var: intp_f(t) for var, intp_f in interpolators.items()},
                    )

            # the update function interpolates the value of input at every
            # step `t`
            def update(t_eval, states):
                if verbose:
                    pbar.n = int((t_eval - t_long[0]) // d_t)
                    pbar.update()
                d_states = np.vstack(
                    self.ode(
                        t=t_eval,
                        states=np.reshape(states, state_var_shape),
                        **{
                            var: intp_f(t_eval) for var, intp_f in interpolators.items()
                        },
                    )
                )
                return d_states.ravel()

        else:  # no external input
            jacc_f = self.jacobian

            # if no current is provided, the system solves ode as defined
            def update(t_eval, states):
                if verbose:
                    pbar.n = int((t_eval - t_long[0]) // d_t)
                    pbar.update()
                d_states = np.vstack(
                    self.ode(states=np.reshape(states, state_var_shape), t=t_eval)
                )
                return d_states.ravel()

        # solver system
        ode_res_info = None
        res = np.zeros((len(t_long), len(self.state_arr), self.num))
        if isinstance(solver, OdeSolver):
            rtol = solver_kws.pop("rtol", 1e-8)
            ode_res = solve_ivp(
                update,
                t_span=(t_long.min(), t_long.max()),
                y0=x0,
                t_eval=t_long,
                method=solver,
                rtol=rtol,
                jac=jacc_f,
            )
            ode_res_info = ode_res
            res = ode_res.y.reshape((len(self.state_arr), -1, len(t_long)))
        elif solver.lower() in ["lsoda", "odeint"]:
            ode_res = odeint(
                update,
                y0=x0,
                t=t_long,
                tfirst=True,
                full_output=full_output,
                Dfun=jacc_f,
                **solver_kws,
            )
            if full_output:
                ode_res_y = ode_res[0]
                ode_res_info = ode_res[1]
                res = ode_res_y.T.reshape((len(self.state_arr), -1, len(t_long)))
            else:
                res = ode_res.T.reshape((len(self.state_arr), -1, len(t_long)))
        else:  # any IVP solver
            rtol = solver_kws.pop("rtol", 1e-8)
            options = {"rtol": rtol}
            if solver.lower() in IVP_SOLVER_WITH_JACC:
                options["jac"] = jacc_f
            ode_res = solve_ivp(
                update,
                t_span=(t_long.min(), t_long.max()),
                y0=x0,
                t_eval=t_long,
                method=solver,
                **options,
            )
            ode_res_info = ode_res
            res = ode_res.y.reshape((len(self.state_arr), -1, len(t_long)))

        res = OrderedDict({key: res[n] for n, key in enumerate(self.states.keys())})

        if verbose:
            pbar.update()
            pbar.close()

        if full_output:
            return res, ode_res_info
        return res

    def clip(self) -> None:
        """Clip state variables based on bounds"""
        for var_name, var_val in self.states.items():
            if self.bounds[var_name] is None:
                continue
            self.states[var_name] = np.clip(
                var_val, self.bounds[var_name][0], self.bounds[var_name][1]
            )
