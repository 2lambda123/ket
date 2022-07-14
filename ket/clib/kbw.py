from ctypes import *
from .wrapper import load_lib
from os import environ

DENSE = 0
SPARSE = 1

API_argtypes = {
    'kbw_run_and_set_result': ([c_void_p, c_int32], []),
    'kbw_run_serialized': ([POINTER(c_uint8), c_size_t, POINTER(c_uint8), c_size_t, c_int32, c_int32], [c_void_p]),
    'kbw_result_get': ([c_void_p], [POINTER(c_uint8), c_size_t]),
    'kbw_result_delete': ([c_void_p], []),
}


def kbw_path():
    from os.path import dirname

    if "KBW_PATH" in environ:
        kbw_path = environ["KBW_PATH"]
    else:
        kbw_path = dirname(__file__)+"/libs/libkbw.so"

    return kbw_path


API = load_lib('KBW', kbw_path(), API_argtypes, 'kbw_error_message')

_sim_mode = None


def set_sim_mode_dense():
    global _sim_mode
    _sim_mode = DENSE


def set_sim_mode_sparse():
    global _sim_mode
    _sim_mode = SPARSE


def run_and_set_result(process):
    global _sim_mode

    if _sim_mode is None and 'KBW_MODE' in environ:
        sim_mode = environ['KBW_MODE'].upper()
        if sim_mode == 'DENSE':
            sim_mode = DENSE
        elif sim_mode == 'SPARSE':
            sim_mode = SPARSE
        else:
            raise RuntimeError(
                "undefined value for environment variable 'KBW_MODE', expecting 'DENSE' or 'SPARSE'")
    elif _sim_mode is not None:
        sim_mode = _sim_mode
    else:
        sim_mode = SPARSE

    API['kbw_run_and_set_result'](process, sim_mode)