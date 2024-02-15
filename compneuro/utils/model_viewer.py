WeChat: cstutorcs
QQ: 749389476
Email: tutorcs@163.com
"""Provides Model Viewer Class that pretty-prints BaseModel content interactively"""
from IPython.display import display, HTML, Code
import numpy as np
import inspect
from traitlets import HasTraits, Int, observe
import ipywidgets as widgets


class ModelViewer(HasTraits):
    neuron_idx = Int(default_value=0)  # index of the neuron to show in model
    model = None  # model for reference
    states_output = widgets.Output()
    params_output = widgets.Output()
    code_output = widgets.Output()
    callbacks_output = None
    refresh_button = None
    neuron_selector = None
    children = None  # a dictionary of elements that is rendered in a tab
    tab = None  # tab instance

    def __init__(self, model: "compneuro.base_model.BaseModel", *args, **kwargs):
        """Model Viewer Shows BaseModel Interactively

        Model viewer creates an interactive view of the basemodel that helps
        user to view the source code and parameter settings of the model.

        The model inherits from :py:func:`traitlets.HasTraits` and takes only 1
        additional positional argument `model` which is an instance of the BaseModel
        to view. For all other arguments and keyword arguments refer to
        :py:func:`traitlets.HasTraits` documentation.

        Arguments:
            model: instance of CompNeuro model to view

        Examples:

            >>> from compneuro.neurons import HodgkinHuxley
            >>> from compneuro.utils.model_viewer import ModelViewer
            >>> hhn = HodgkinHuxley()
            >>> viewer = ModelViewer(hhn) # or just ModelViewer(hhn)
            >>> viewer
        """
        super().__init__(*args, **kwargs)
        self.model = model
        with self.states_output:
            self.states_output.clear_output()
            display(self.create_states_table(self.neuron_idx))
        with self.params_output:
            self.params_output.clear_output()
            display(self.create_params_table(self.neuron_idx))
        with self.code_output:
            self.code_output.clear_output()
            display(Code(inspect.getsource(self.model.ode), language="Python"))

        callback_code_blocks = {}
        for n, func in enumerate(model.callbacks):
            output = widgets.Output()
            with output:
                display(Code(inspect.getsource(func), language="Python"))
            name = func.__name__ if func.__name__ != "<lambda>" else f"Callback-{n}"
            callback_code_blocks[name] = output
        self.callbacks_output = widgets.Accordion(
            children=list(callback_code_blocks.values())
        )
        for i, ttl in enumerate(callback_code_blocks.keys()):
            self.callbacks_output.set_title(i, ttl)

        self.refresh_button = widgets.Button(
            description="",
            disabled=False,
            button_style="",  # 'success', 'info', 'warning', 'danger' or ''
            tooltip="Description",
            icon="refresh",
        )
        self.neuron_selector = widgets.BoundedIntText(
            min=0,
            max=self.model.num - 1,
            value=0,
            description=f"Neuron Idx ({0}-{self.model.num-1})",
            continuous_update=False,
        )

        def _neuron_selector_callback(change):
            self.neuron_idx = change["new"]

        self.neuron_selector.observe(
            lambda change: _neuron_selector_callback(change), names="value"
        )
        self.refresh_button.on_click(lambda b: self.refresh())
        self.children = dict(
            Code=self.code_output,
            Callbacks=self.callbacks_output,
            Equation=widgets.HTMLMath(self.model._parsed_model.pprint().data),
            States=widgets.VBox(
                [
                    widgets.HBox([self.neuron_selector, self.refresh_button]),
                    self.states_output,
                ]
            ),
            Parameters=widgets.VBox(
                [
                    widgets.HBox([self.neuron_selector, self.refresh_button]),
                    self.params_output,
                ]
            ),
        )
        self.tab = widgets.Tab()
        self.tab.children = list(self.children.values())
        for i, ttl in enumerate(self.children.keys()):
            self.tab.set_title(i, ttl)

    @observe("neuron_idx")
    def _observe_neuron_idx(self, change):
        """Refresh view of neuron index changed"""
        if change["new"] == change["old"]:
            return
        self.refresh()

    def _repr_html_(self):
        """Pretty print in html environment"""
        return display(self.tab)

    def __repr__(self):
        return f"ModelViewer({self.model.__class__})"

    def refresh(self):
        """Manually refresh content, only changes states and parameters"""
        with self.states_output:
            self.states_output.clear_output()
            display(self.create_states_table(self.neuron_idx))
        with self.params_output:
            self.params_output.clear_output()
            display(self.create_params_table(self.neuron_idx))

    def create_states_table(self, idx):
        """Helper function to create states table based on neuron `idx`"""
        values_arr = [str(np.atleast_1d(p)[idx]) for p in self.model.states.values()]
        names_arr = [name for name in self.model.states.keys()]
        initials_arr = [
            str(np.atleast_1d(p)[idx]) for p in self.model.initial_states.values()
        ]
        lb_arr = [
            str(p[0] if p is not None else None) for p in self.model.bounds.values()
        ]
        ub_arr = [
            str(p[1] if p is not None else None) for p in self.model.bounds.values()
        ]
        header = "<th></th><th>" + "</th><th>".join(names_arr) + "</th>"
        values = "<td><i>Value</i></td><td>" + "</td><td>".join(values_arr) + "</td>"
        initials = (
            "<td><i>Initial Value</i></td><td>"
            + "</td><td>".join(initials_arr)
            + "</td>"
        )
        lower_bounds = (
            "<td><i>Lower Bound</i></td><td>" + "</td><td>".join(lb_arr) + "</td>"
        )
        upper_bounds = (
            "<td><i>Upper Bound</i></td><td>" + "</td><td>".join(ub_arr) + "</td>"
        )
        table = "<table><tr>{}</tr><tr>{}</tr><tr>{}</tr><tr>{}</tr><tr>{}</tr></table>".format(
            header, values, initials, lower_bounds, upper_bounds
        )
        return HTML(table)

    def create_params_table(self, idx):
        """Helper function to create parameters table based on neuron `idx`"""
        names_arr = [name for name in self.model.params.keys()]
        values_arr = [
            str(p) if np.isscalar(p) else str(np.atleast_1d(p)[idx])
            for p in self.model.params.values()
        ]

        header = "</th><th>".join(names_arr)
        values = "</td><td>".join(values_arr)
        table = "<table><tr><th>{header}</th></tr><tr><td>{values}</td></tr></table>".format(
            header=header, values=values
        )
        return HTML(table)
