"""Auto-load activation classes for convenient imports.

Any new module in this package that defines subclasses of
`LayerBase` will automatically be imported into the package
namespace.
"""

from .base import LayerBase

import importlib
import inspect
import pkgutil

__all__ = []

for module_info in pkgutil.iter_modules(__path__):
    if module_info.ispkg or module_info.name == "base":
        continue
    module = importlib.import_module(f"{__name__}.{module_info.name}")
    for attribute_name, attribute in vars(module).items():
        if (
            inspect.isclass(attribute)
            and issubclass(attribute, LayerBase)
            and attribute is not LayerBase
        ):
            globals()[attribute_name] = attribute
            __all__.append(attribute_name)