"""Async-first Harness application facade and composition root.

Public symbols are loaded lazily so importing a narrow application service does not
import the runtime/composition dependency graph.
"""

from importlib import import_module

_EXPORTS = {
    "Agent": ("src.application.agent", "Agent"),
    "AgentSession": ("src.application.agent", "AgentSession"),
    "ApplicationConfig": ("src.application.composition", "ApplicationConfig"),
    "create_development_agent": (
        "src.application.composition",
        "create_development_agent",
    ),
    "create_production_agent": (
        "src.application.composition",
        "create_production_agent",
    ),
    "ApplicationClosedError": (
        "src.application.runtime",
        "ApplicationClosedError",
    ),
    "RuntimeBusyError": ("src.application.runtime", "RuntimeBusyError"),
    "RuntimeCoordinator": ("src.application.runtime", "RuntimeCoordinator"),
    "SessionCapacityError": ("src.application.runtime", "SessionCapacityError"),
    "SessionOwnershipError": ("src.application.runtime", "SessionOwnershipError"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> object:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError:
        raise AttributeError(name) from None
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
