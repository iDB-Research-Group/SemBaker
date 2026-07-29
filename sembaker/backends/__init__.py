"""Engine adapters. Import the one matching the engine you use.

Each adapter module imports its engine at import time; if the engine is not
installed you get an ImportError telling you which extra to install.

    import sembaker.backends.pz as cxpz          # Palimpzest   sembaker[pz]
    import sembaker.backends.lotus as cxlotus    # LOTUS        sembaker[lotus]
    import sembaker.backends.nirvana as cxnv     # Nirvana      sembaker[nirvana]
    import sembaker.backends.docetl as cxdoc     # DocETL       sembaker[docetl]

`sembaker.backends.pz_ops` holds the Compiled* physical operators + rules that
the PZ adapter registers into Palimpzest's optimizer at runtime.
"""

import importlib

_ADAPTERS = {"pz", "lotus", "nirvana", "docetl", "pz_ops"}


def __getattr__(name: str):
    if name in _ADAPTERS:
        return importlib.import_module(f"sembaker.backends.{name}")
    raise AttributeError(f"module 'sembaker.backends' has no attribute {name!r}")
