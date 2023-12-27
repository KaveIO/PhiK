import importlib.util

_ext_spec = importlib.util.find_spec("phik.lib._phik_simulation_core")
if _ext_spec is not None:
    from phik.lib._phik_simulation_core import _sim_2d_data_patefield

    CPP_SUPPORT = True
else:
    CPP_SUPPORT = False

    def _sim_2d_data_patefield(*args, **kwargs):
        msg = "Patefield requires a compiled extension that was not found."
        raise NotImplementedError(msg)


__all__ = ["CPP_SUPPORT", "_sim_2d_data_patefield"]
