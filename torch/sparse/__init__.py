# The Tensor classes are added to this module by python_tensor.cpp
from typing import Optional, Tuple, List, Union

import torch
from torch import Tensor

# A workaround to support both TorchScript and MyPy:
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from torch.types import _dtype as DType
    DimOrDims = Optional[Union[int, Tuple[int], List[int]]]
else:
    # The JIT doesn't understand Union, nor torch.dtype here
    DType = int
    DimOrDims = Optional[Tuple[int]]


__all__ = [
    'addmm',
    'mm',
    'sum',
    'softmax',
    'log_softmax',
]


def addmm(mat: Tensor, mat1: Tensor, mat2: Tensor,
          beta: float = 1., alpha: float = 1.) -> Tensor:
    r"""
    This function does exact same thing as :func:`torch.addmm` in the forward,
    except that it supports backward for sparse matrix :attr:`mat1`. :attr:`mat1`
    need to have `sparse_dim = 2`. Note that the gradients of :attr:`mat1` is a
    coalesced sparse tensor.

    Args:
        mat (Tensor): a dense matrix to be added
        mat1 (Tensor): a sparse matrix to be multiplied
        mat2 (Tensor): a dense matrix to be multiplied
        beta (Number, optional): multiplier for :attr:`mat` (:math:`\beta`)
        alpha (Number, optional): multiplier for :math:`mat1 @ mat2` (:math:`\alpha`)
    """
    return torch._sparse_addmm(mat, mat1, mat2, beta=beta, alpha=alpha)


def mm(mat1: Tensor, mat2: Tensor) -> Tensor:
    r"""
    Performs a matrix multiplication of the sparse matrix :attr:`mat1`
    and the (sparse or strided) matrix :attr:`mat2`. Similar to :func:`torch.mm`, If :attr:`mat1` is a
    :math:`(n \times m)` tensor, :attr:`mat2` is a :math:`(m \times p)` tensor, out will be a
    :math:`(n \times p)` tensor. :attr:`mat1` need to have `sparse_dim = 2`.
    This function also supports backward for both matrices. Note that the gradients of
    :attr:`mat1` is a coalesced sparse tensor.

    Args:
        mat1 (SparseTensor): the first sparse matrix to be multiplied
        mat2 (Tensor): the second matrix to be multiplied, which could be sparse or dense

    Shape:
        The format of the output tensor of this function follows:
        - sparse x sparse -> sparse
        - sparse x dense -> dense

    Example::

        >>> a = torch.randn(2, 3).to_sparse().requires_grad_(True)
        >>> a
        tensor(indices=tensor([[0, 0, 0, 1, 1, 1],
                               [0, 1, 2, 0, 1, 2]]),
               values=tensor([ 1.5901,  0.0183, -0.6146,  1.8061, -0.0112,  0.6302]),
               size=(2, 3), nnz=6, layout=torch.sparse_coo, requires_grad=True)

        >>> b = torch.randn(3, 2, requires_grad=True)
        >>> b
        tensor([[-0.6479,  0.7874],
                [-1.2056,  0.5641],
                [-1.1716, -0.9923]], requires_grad=True)

        >>> y = torch.sparse.mm(a, b)
        >>> y
        tensor([[-0.3323,  1.8723],
                [-1.8951,  0.7904]], grad_fn=<SparseAddmmBackward>)
        >>> y.sum().backward()
        >>> a.grad
        tensor(indices=tensor([[0, 0, 0, 1, 1, 1],
                               [0, 1, 2, 0, 1, 2]]),
               values=tensor([ 0.1394, -0.6415, -2.1639,  0.1394, -0.6415, -2.1639]),
               size=(2, 3), nnz=6, layout=torch.sparse_coo)
    """
    if mat1.is_sparse and mat2.is_sparse:
        return torch._sparse_sparse_matmul(mat1, mat2)
    return torch._sparse_mm(mat1, mat2)


def sum(input: Tensor, dim: DimOrDims = None,
        dtype: Optional[DType] = None) -> Tensor:
    r"""
    Returns the sum of each row of the sparse tensor :attr:`input` in the given
    dimensions :attr:`dim`. If :attr:`dim` is a list of dimensions,
    reduce over all of them. When sum over all ``sparse_dim``, this method
    returns a dense tensor instead of a sparse tensor.

    All summed :attr:`dim` are squeezed (see :func:`torch.squeeze`), resulting an output
    tensor having :attr:`dim` fewer dimensions than :attr:`input`.

    During backward, only gradients at ``nnz`` locations of :attr:`input`
    will propagate back. Note that the gradients of :attr:`input` is coalesced.

    Args:
        input (Tensor): the input sparse tensor
        dim (int or tuple of ints): a dimension or a list of dimensions to reduce. Default: reduce
            over all dims.
        dtype (:class:`torch.dtype`, optional): the desired data type of returned Tensor.
            Default: dtype of :attr:`input`.

    Example::

        >>> nnz = 3
        >>> dims = [5, 5, 2, 3]
        >>> I = torch.cat([torch.randint(0, dims[0], size=(nnz,)),
                           torch.randint(0, dims[1], size=(nnz,))], 0).reshape(2, nnz)
        >>> V = torch.randn(nnz, dims[2], dims[3])
        >>> size = torch.Size(dims)
        >>> S = torch.sparse_coo_tensor(I, V, size)
        >>> S
        tensor(indices=tensor([[2, 0, 3],
                               [2, 4, 1]]),
               values=tensor([[[-0.6438, -1.6467,  1.4004],
                               [ 0.3411,  0.0918, -0.2312]],

                              [[ 0.5348,  0.0634, -2.0494],
                               [-0.7125, -1.0646,  2.1844]],

                              [[ 0.1276,  0.1874, -0.6334],
                               [-1.9682, -0.5340,  0.7483]]]),
               size=(5, 5, 2, 3), nnz=3, layout=torch.sparse_coo)

        # when sum over only part of sparse_dims, return a sparse tensor
        >>> torch.sparse.sum(S, [1, 3])
        tensor(indices=tensor([[0, 2, 3]]),
               values=tensor([[-1.4512,  0.4073],
                              [-0.8901,  0.2017],
                              [-0.3183, -1.7539]]),
               size=(5, 2), nnz=3, layout=torch.sparse_coo)

        # when sum over all sparse dim, return a dense tensor
        # with summed dims squeezed
        >>> torch.sparse.sum(S, [0, 1, 3])
        tensor([-2.6596, -1.1450])
    """
    if dtype is None:
        if dim is not None:
            return torch._sparse_sum(input, dim)
        else:
            return torch._sparse_sum(input)
    else:
        if dim is not None:
            return torch._sparse_sum(input, dim, dtype=dtype)
        else:
            return torch._sparse_sum(input, dtype=dtype)


def softmax(input: Tensor, dim: int, dtype: Optional[DType] = None) -> Tensor:
    r"""Applies a softmax function.

    Softmax is defined as:

    :math:`\text{Softmax}(x_{i}) = \frac{exp(x_i)}{\sum_j exp(x_j)}`

    where :math:`i, j` run over sparse tensor indices and unspecified
    entries are ignores. This is equivalent to defining unspecified
    entries as negative infinity so that :math:`exp(x_k) = 0` when the
    entry with index :math:`k` has not specified.

    It is applied to all slices along `dim`, and will re-scale them so
    that the elements lie in the range `[0, 1]` and sum to 1.

    Args:
        input (Tensor): input
        dim (int): A dimension along which softmax will be computed.
        dtype (:class:`torch.dtype`, optional): the desired data type
          of returned tensor.  If specified, the input tensor is
          casted to :attr:`dtype` before the operation is
          performed. This is useful for preventing data type
          overflows. Default: None
    """
    return torch._sparse_softmax(input, dim, dtype=dtype)


def log_softmax(input: Tensor, dim: int, dtype: Optional[DType] = None) -> Tensor:
    r"""Applies a softmax function followed by logarithm.

    See :class:`~torch.sparse.softmax` for more details.

    Args:
        input (Tensor): input
        dim (int): A dimension along which softmax will be computed.
        dtype (:class:`torch.dtype`, optional): the desired data type
          of returned tensor.  If specified, the input tensor is
          casted to :attr:`dtype` before the operation is
          performed. This is useful for preventing data type
          overflows. Default: None
    """
    return torch._sparse_log_softmax(input, dim, dtype=dtype)


# All masked reduction/normalization operations have the same
# signatures. Here we introduce docstring templates that are applied
# to docstrings of reduction/normalization functions via
# _apply_docstring_templates decorator.

def _apply_docstring_templates(func):
    """Decorator that applies docstring templates to function docstring
    and returns the function instance.
    """

    docstring_templates = dict(
        masked_reduction_signature='''\
{function_name}(input, dim, keepdim=False, dtype=None, mask=None) -> Tensor''',
        masked_reduction_descr='''\
Returns {operation name} of :attr:`input` tensor along given dimension
:attr:`dim` while the :attr:`input` elements are masked out according
to the boolean tensor :attr:`mask`.''',
        masked_reduction_args='''\
If :attr:`keepdim` is ``True``, the output tensor is of the same
size as :attr:`input` except in the dimension(s) :attr:`dim` where
it is of size 1.  Otherwise, :attr:`dim` is squeezed (see
:func:`torch.squeeze`), resulting in the output tensor having 1
fewer dimension.

The boolean tensor :attr:`mask` defines the "validity" of
:attr:`input` tensor elements: if :attr:`mask` element is True
then the corresponding element in :attr:`input` tensor will be
included in {function name} computation, otherwise the element is
ignored.

When all elements of :attr:`input` along the given dimension
:attr:`dim` are ignored, the corresponding element of the output
tensor will have undefined value: it may or may not correspond to the
identity value of {operation name} operation; the choice may
correspond to the value that leads to the most efficient storage of
:attr:`output` tensor.

The shapes of the :attr:`mask` tensor and the :attr:`input` tensor
don't need to match, but they must be :ref:`broadcastable
<broadcasting-semantics>`.

Args:
    input (Tensor): the input tensor
    dim (int): a dimension along which {function name} will be
      computed.

Keyword args:
    keepdim (bool, optional): whether the output tensor has
      :attr:`dim` retained or not. Default: False.
    dtype (:class:`torch.dtype`, optional): the desired data type
      of returned tensor.  If specified, the input tensor is
      casted to :attr:`dtype` before the operation is
      performed. Default: None
    mask (:class:`torch.Tensor`, optional)_reduction_identity(
                 'sparse.' + func.__name__,
                 torch.tensor(0, dtype=torch.int16)): the boolean tensor
      containing the binary mask of validity of input tensor
      elements.
      Default: ``torch.ones(input.shape, dtype=torch.bool)``.''',
        masked_reduction_example='''\
Example::

    >>> input = torch.tensor([[-3, -2, -1], [0, 1, 2]])
    >>> input
    tensor([[-3, -2, -1],
            [ 0,  1,  2]])
    >>> mask = torch.tensor([[True, False, True], [False, False, False]])
    >>> mask
    tensor([[ True, False,  True],
            [False, False, False]])''',
        masked_reduction_identity='''\
The identity value of {function name}, which is used to start the reduction, is {identity_int32}.''',
        masked_reduction_identity_dtype='''\
The identity value of {function name}, which is used to start the
reduction, depends on input dtype. For instance, for float32, uint8,
and int32 dtypes, the identity values are {identity_float32}, {identity_uint8}, and {identity_int32}, respecitively.''',
        masked_normalization_args='''\
    TBD
''')

    # Apply function name info to docstring templates:
    templates = dict(
        (k, v.format_map(
            {'function_name': func.__name__,
             'function name': func.__name__.replace('_', ' '),
             'operation_name': '_'.join(func.__name__.split('_')[1:]),
             'operation name': ' '.join(func.__name__.split('_')[1:]),
             'identity_uint8': dict(masked_sum='0',
                                    masked_prod='1',
                                    masked_amax=0,
                                    masked_amin=2 ** 8 - 1).get(func.__name__),
             'identity_int32': dict(masked_sum='0',
                                    masked_prod='1',
                                    masked_amax=-2 ** 31,
                                    masked_amin=2 ** 31 - 1).get(func.__name__),
             'identity_float32': dict(masked_sum='0.0',
                                      masked_prod='1.0',
                                      masked_amax='-inf',
                                      masked_amin='inf').get(func.__name__)}
        )) for k, v in docstring_templates.items())

    # Apply docstring templates to function doctring:
    func.__doc__ = func.__doc__.format_map(templates)

    # Expose function as public symbol
    __all__.append(func.__name__)

    return func


def _reduction_identity(op_name: str, input: Tensor):
    """Return identity value as scalar tensor of a reduction operation on
    given input.
    """
    dtype: DType = input.dtype
    device = input.device
    if op_name == 'sparse.masked_sum':
        return torch.tensor(0, dtype=dtype, device=device)
    elif op_name == 'sparse.masked_prod':
        return torch.tensor(1, dtype=dtype, device=device)
    elif op_name == 'sparse.masked_amax':
        if torch.is_floating_point(input):
            return torch.tensor(-torch.inf, dtype=dtype, device=device)
        elif torch.is_signed(input):
            return torch.tensor(torch.iinfo(dtype).min, dtype=dtype, device=device)
        elif dtype == torch.uint8:
            return torch.tensor(0, dtype=dtype, device=device)
    elif op_name == 'sparse.masked_amin':
        if torch.is_floating_point(input):
            return torch.tensor(torch.inf, dtype=dtype, device=device)
        elif torch.is_signed(input):
            return torch.tensor(torch.iinfo(dtype).max, dtype=dtype, device=device)
        elif dtype == torch.uint8:
            return torch.tensor(255, dtype=dtype, device=device)
    raise NotImplementedError(f'identity of {op_name} on {dtype} input')


def _canonical_dim(dim: DimOrDims, ndim: int) -> Tuple[int, ...]:
    """Return dim argument as a tuple of sorted dim values.
    """
    dims: List[int] = []
    if dim is None:
        return tuple(range(ndim))
    ndim = max(ndim, 1)
    dim_ = (dim,) if isinstance(dim, int) else dim
    for d in dim_:
        if d in dims:
            raise RuntimeError(f'dim={d} appears multiple times in the list of dims')
        if d >= ndim or d < -ndim:
            raise IndexError(f'Dimension out of range (expected to be in range of [{-ndim}, {ndim-1}], but got {d})')
        dims.append(d % ndim)
    return tuple(sorted(dims))


@_apply_docstring_templates
def masked_sum(input: Tensor,
               dim: DimOrDims = None,
               *,
               keepdim: Optional[bool] = False,
               dtype: Optional[DType] = None,
               mask: Optional[Tensor] = None) -> Tensor:
    """
{masked_reduction_signature}

{masked_reduction_descr}

{masked_reduction_identity}

{masked_reduction_args}

{masked_reduction_example}
    >>> torch.sparse.masked_sum(input, 1, mask=mask)
    tensor([-4,  0])
    """
    if dtype is None:
        dtype = input.dtype
    # TODO: What follows is a reference implementation of a masked sum
    # operation that is to be replaced with an optimized one and
    # extended to support other layouts.
    if input.layout == torch.strided:
        mask_input = input if mask is None else torch.where(mask, input, torch.zeros_like(input))
        if dim is None:
            return torch.sum(mask_input, dtype=dtype)
        else:
            return torch.sum(mask_input, dim, bool(keepdim), dtype=dtype)
    else:
        raise NotImplementedError(f'masked_sum of {input.layout} tensor')


@_apply_docstring_templates
def masked_amax(input: Tensor,
                dim: DimOrDims = None,
                *,
                keepdim: Optional[bool] = False,
                dtype: Optional[DType] = None,
                mask: Optional[Tensor] = None) -> Tensor:
    """
{masked_reduction_signature}

{masked_reduction_descr}

{masked_reduction_identity_dtype}

{masked_reduction_args}

{masked_reduction_example}
    >>> torch.sparse.masked_amax(input, 1, mask=mask)
    tensor([                  -1, -9223372036854775808])
    """
    if dtype is None:
        dtype = input.dtype
    if input.layout == torch.strided:
        if mask is None:
            mask_input = input
        else:
            identity = torch.empty_like(input)
            identity.fill_(_reduction_identity('sparse.masked_amax', input))
            mask_input = torch.where(mask, input, identity)
        dim_ = _canonical_dim(dim, mask_input.ndim)
        return torch.amax(mask_input, dim_, bool(keepdim)).to(dtype=dtype)
    else:
        raise NotImplementedError(f'masked_amax of {input.layout} tensor')


def _output_mask(input: Tensor,
                 dim: DimOrDims = None,
                 *,
                 keepdim: Optional[bool] = False,
                 dtype: Optional[DType] = None,  # unused
                 mask: Optional[Tensor] = None) -> Tensor:
    """Return the output mask of an masked reduction operation.

    This function is equivalent to torch.any with broadcasting mask to
    input shape and supporting multiple dims. Used to implement masked
    equality check required in testing masked reductions.
    """
    if mask is None:
        if input.layout == torch.strided:
            outmask = torch.ones(input.shape, dtype=torch.bool, device=input.device)
        else:
            raise NotImplementedError(f'mask from layout {input.layout}')
    elif mask.ndim < input.ndim:
        outmask = torch.broadcast_to(mask.clone(), input.shape).to(dtype=torch.bool)
    elif mask.ndim > input.ndim:
        raise NotImplementedError("mask dimensionality higher than of input")
    else:
        outmask = mask.to(dtype=torch.bool)
    if isinstance(dim, tuple):
        # Workaround https://github.com/pytorch/pytorch/issues/56586
        for d in reversed(_canonical_dim(dim, input.ndim)):
            outmask = outmask.any(dim=d, keepdim=bool(keepdim))
    elif isinstance(dim, int):
        outmask = outmask.any(dim=dim, keepdim=bool(keepdim))
    else:
        assert dim is None
        outmask = outmask.any()
    return outmask
