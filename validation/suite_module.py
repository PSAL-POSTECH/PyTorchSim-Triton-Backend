"""One suite's script as a module for another: validation/<suite>/<script>.py, imported as <suite>_<script>
so it never shadows the script that asks."""
import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def load(suite, script):
    """validation/<suite>/<script>.py, executed as the module <suite>_<script>."""
    spec = importlib.util.spec_from_file_location(f"{suite}_{script}", os.path.join(HERE, suite, f"{script}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
