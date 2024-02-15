WeChat: cstutorcs
QQ: 749389476
Email: tutorcs@163.com
"""Comp Neuro Exception Classes

.. note::

    All anticipated exceptions should raise custom Exceptions so that user
    can identify when an error is not anticipated and help with debugging.
"""


class CompNeuroError(Exception):
    """Base Comp Neuro Exceptions"""


class CompNeuroUtilsError(CompNeuroError):
    """Base Comp Neuro Utility Exceptions"""


class CompNeuroPlotError(CompNeuroUtilsError):
    """Comp Neuro Plotter Exceptions"""


class CompNeuroSignalError(CompNeuroUtilsError):
    """Comp Neuro Signal Exceptions"""


class CompNeuroModelError(CompNeuroError):
    """Comp Neuro Model Exceptions"""


class CompNeuroWarning(Warning):
    """Base Comp Neuro Warning"""


class CompNeuroUtilsWarning(CompNeuroWarning):
    """Base Comp Neuro Utility Warnings"""


class CompNeuroPlotWarning(CompNeuroUtilsWarning):
    """Comp Neuro Plotter Warnings"""


class CompNeuroModelFunctionDisabledError(CompNeuroModelError):
    """Exception raised when user attempt to execute an explicitly disabled function"""
