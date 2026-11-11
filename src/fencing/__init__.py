"""Humanoid Combat Fencing glue code.

Pipeline (Goal 1): scrape -> screen/isolate -> pose -> retarget -> motion cleanup
-> train (mjlab) -> evaluate -> sim2sim. Each stage is a subcommand of `fencing`
(see fencing.cli). Heavy stages run inside their upstream project's own venv;
fencing.paths knows where those live.
"""

__version__ = "0.1.0"
