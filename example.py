import torch
from torch.utils._pytree import tree_map, tree_flatten
from torch.utils._python_dispatch import enable_python_mode
from functools import partial
import contextlib

supported_ops = {
    torch.ops.aten.zeros,
    torch.ops.aten.mul,
    torch.ops.aten.select,
    torch.ops.aten.randn,
    torch.ops.aten.sum,
    torch.ops.aten.sin,
    torch.ops.aten.add,
    torch.ops.aten.stack,
    torch.ops.aten.to,
    torch.ops.aten.detach,
    torch.ops.aten.expand,
}

@contextlib.contextmanager
def no_dispatch():
    guard = torch._C._DisableTorchDispatch()  # type: ignore[attr-defined]
    try:
        yield
    finally:
        del guard

def pop_mode_stack():
    return torch._C._autograd._exit_python_mode()

def push_mode_stack(subclass, mode):
    return torch._C._autograd._enter_python_mode(subclass, mode)

def mode_stack_size():
    return torch._C._autograd._mode_stack_size()

@contextlib.contextmanager
def temporarily_pop_mode_stack():
    assert mode_stack_size() > 0
    subclass, mode = pop_mode_stack()
    try:
        yield subclass, mode
    finally:
        push_mode_stack(subclass, mode)

@contextlib.contextmanager
def noop():
    yield None, None

def batched_fallback(func, subclass, args, kwargs):
    if not func in supported_ops:
        raise RuntimeError(f"not supported: {func.__name__}")

    # TODO: assumes there are no Tensors in kwargs
    flat_args, _ = tree_flatten(args)
    if not any([isinstance(e, subclass) for e in flat_args]):
        return func(*args)

    # Naive batching rule: for-loop + stack
    bdim_size = None
    for e in flat_args:
        if isinstance(e, subclass):
            bdim_size = e.elem.size(e.bdim)

    def get_slice(idx, e):
        return e.elem.select(e.bdim, idx) if isinstance(e, subclass) else e

    results = []
    for i in range(bdim_size):
        sliced_args = tree_map(partial(get_slice, i), args)
        res = func(*sliced_args, **kwargs)
        assert isinstance(res, torch.Tensor)
        results.append(res)

    result = torch.stack(results)
    return subclass(result, 0)

def get_torch_dispatch(subclass):
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        ctx = temporarily_pop_mode_stack if mode_stack_size() > 0 else noop
        with ctx() as (mode_subclass, mode):
            assert mode_subclass is None or subclass == mode_subclass

            if func == torch.ops.aten.randn and mode is not None:
                if mode.randomness == 'error':
                    raise RuntimeError("No randomness allowed")
                if mode.randomness == 'same':
                    return func(*args, **kwargs)
                if mode.randomness == 'different':
                    args = list(args)
                    args[0] = [mode.batch_size] + args[0]
                    return subclass(func(*args, **kwargs), 0)

            return batched_fallback(func, subclass, args, kwargs)

    return __torch_dispatch__

class BatchedTensor(torch.Tensor):
    elem: torch.Tensor
    __torch_function__ = torch._C._disabled_torch_function_impl

    __slots__ = ['elem', 'bdim']

    @staticmethod
    def __new__(cls, elem, bdim, *args, **kwargs):
        r = torch.Tensor._make_subclass(cls, elem.to('cpu')[0], elem.requires_grad)
        r.elem = elem
        r.bdim = bdim
        r.current_subclass = BatchedTensor
        return r

    def __repr__(self):
        return f"BatchedTensor({self.elem}, {self.bdim})"

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        return get_torch_dispatch(BatchedTensor)(cls, func, types, args, kwargs)

def gen_batchedtensor_subclass():
    # Generate a fresh new class on the fly
    class GeneratedBatchedTensor(BatchedTensor):
        @classmethod
        def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
            return get_torch_dispatch(GeneratedBatchedTensor)(cls, func, types, args, kwargs)
    return GeneratedBatchedTensor

class VmapMode():
    def __init__(self, batch_size, randomness):
        self.batch_size = batch_size
        self.randomness = randomness

    def __enter__(self):
        subclass = gen_batchedtensor_subclass()
        push_mode_stack(subclass, self)
        return subclass

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pop_mode_stack()

# randomness = {"error", "same", "different"}
def vmap(f, in_dims=(0,), randomness="error"):
    def wrapped(*args):
        batch_sizes = [arg.size(in_dim) for in_dim, arg in zip(in_dims, args) if in_dim is not None]
        batch_size = batch_sizes[0]

        with VmapMode(batch_size, randomness) as GenBatchedTensor:
            def wrap(e, in_dim):
                assert in_dim is None or in_dim == 0
                if in_dim is None:
                    return e
                return GenBatchedTensor(e, in_dim)

            batched_args = tuple(wrap(arg, in_dim) for arg, in_dim in zip(args, in_dims))
            batched_out = f(*batched_args)
            assert isinstance(batched_out, torch.Tensor)

        if isinstance(batched_out, GenBatchedTensor):
            return batched_out.elem
        else:
            return batched_out.expand(batch_size, *batched_out.shape)
    return wrapped

# basic vmap test
x = torch.randn(3, 2, 5, 7)
y = vmap(vmap(vmap(torch.sum)))(x)
assert torch.allclose(y, x.sum([-1]))

# complicated vmap test
x = torch.arange(3)
y = torch.arange(4)
z = vmap(vmap(torch.mul, (0, None)), (None, 0))(x, y)
assert torch.allclose(z, y.unsqueeze(-1) * x)

# vmap mode test
def foo(x):
    return torch.randn(1)
y = vmap(foo, randomness='same')(torch.ones(3))
assert torch.allclose(y - y[0], torch.zeros_like(y))
z = vmap(foo, randomness='different')(torch.ones(3))
assert not torch.allclose(z - z[0], torch.zeros_like(z))

# TODO: figure out jvp.
# can we use pytorch dual tensors? probably not...

# 
# BatchedTensor0 = gen_batchedtensor_subclass()
# BatchedTensor1 = gen_batchedtensor_subclass()
# 
# x = torch.randn(3)
# y = torch.randn(4)
# Bx = BatchedTensor0(x, 0)
# By = BatchedTensor1(y, 0)
# 
# with enable_python_mode(BatchedTensor0):
#     with enable_python_mode(BatchedTensor1):
#         Bz = torch.mul(Bx, By)


# 
# x = torch.arange(3)
# y = torch.arange(4)
# z = vmap(vmap(torch.mul, (0, None)), (None, 0))(x, y)
# assert torch.allclose(z, y.unsqueeze(-1) * x)
# 
# # The following needed me to hack something inside PyTorch core.
# # Up until this point, all of the examples were functorch-only :O.
# x = torch.randn(3, 2, 5, 7)
# y = vmap(vmap(torch.sum, (0,)), (0,))(x)
# assert torch.allclose(y, x.sum([-1, -2]))
# 
# x = torch.randn(3, 2, 5, 7)
# y = vmap(vmap(vmap(torch.sum, (0,))), (0,))(x)
# assert torch.allclose(y, x.sum([-1]))
