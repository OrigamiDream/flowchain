import inspect
import tensorflow as tf

from inspect import getmembers, isfunction
from typing import Any, Optional, List, Callable, Tuple, Union

_TENSOR_FN_CACHE = dict()
_DEFAULT_MODULE_WHITELIST = [
    ('tensorflow.python.ops.array_ops', ['!concat', '!fill', '!meshgrid', '!ones', '!zeros', '!stack']),
    ('tensorflow.python.ops.gen_array_ops', ['!tile']),
    'tensorflow.python.ops.gen_math_ops',
    'tensorflow.python.ops.gen_nn_ops',
    'tensorflow.python.ops.gen_string_ops',
    'tensorflow.python.ops.math_ops',
    'tensorflow.python.ops.sort_ops',
    ('tensorflow.python.ops.array_ops', ['diag', 'diag_part']),
    ('tensorflow.python.ops.linalg_ops', ['!eye']),
    ('tensorflow.python.ops.nn_impl', ['l2_normalize', 'normalize']),
]
_DEFAULT_OBJECT_MAPPINGS = [(obj, _DEFAULT_MODULE_WHITELIST) for obj in [tf, tf.linalg, tf.math]]


def _is_allowed(func: Callable, func_name: str, whitelist: List[Union[str, Tuple[str, List[str]]]]):
    module_name = inspect.getmodule(func).__name__
    for condition in whitelist:
        if isinstance(condition, tuple):
            allowed_module, func_names = condition
        else:
            allowed_module = condition
            func_names = None

        if module_name != allowed_module:
            continue

        if func_names is not None:
            pos_fn_names = [name for name in func_names if not name.startswith('!')]
            neg_fn_names = [name for name in func_names if name.startswith('!')]
            if len(pos_fn_names) > 0 and len(neg_fn_names) > 0:
                raise ValueError('Whitelist must be either all positives or all negatives.')

            if len(neg_fn_names) > 0 and func_name in neg_fn_names:
                continue

            if len(pos_fn_names) > 0 and func_name not in pos_fn_names:
                continue
        return True
    return False


def _retrieve_tf_func(obj: Any = tf, module_whitelist: Optional[List[str]] = None):
    def _tf_func_wrap(fn: Callable) -> Callable:
        def _invoke(*args, **kwargs):
            return fn(*args, **kwargs)

        _invoke.__doc__ = fn.__doc__
        _invoke.__name__ = fn.__name__
        _invoke.__signature__ = inspect.signature(fn)
        return _invoke

    tf_funcs = getmembers(obj, isfunction)
    funcs = dict()
    for name, func in tf_funcs:
        if not _is_allowed(func, name, module_whitelist):
            continue

        signature = inspect.signature(func)
        if len(signature.parameters) == 0:
            continue
        funcs[name] = _tf_func_wrap(func)
    return funcs


def register_tensor_chaining(obj, mappings=None):
    global _TENSOR_FN_CACHE

    if mappings is None:
        mappings = _DEFAULT_OBJECT_MAPPINGS

    key = str(obj)
    if key in _TENSOR_FN_CACHE:
        return False
    func_map = dict()
    for orig, whitelist in mappings:
        func_map.update(_retrieve_tf_func(orig, whitelist))
    for tf_name, tf_func in func_map.items():
        if hasattr(obj, tf_name) or 'type' == tf_name:
            continue
        setattr(obj, tf_name, tf_func)
    _TENSOR_FN_CACHE[key] = func_map
    return True


def enable_tensor_chaining(mappings=None):
    register_tensor_chaining(tf.Variable, mappings)
    register_tensor_chaining(tf.Tensor, mappings)
    register_tensor_chaining(tf.RaggedTensor, mappings)
    register_tensor_chaining(tf.SparseTensor, mappings)
