import torch
from torch import Tensor


def detect_batch_size(inputs, batch_dims):
    batch_size = None
    for x, x_bdim in zip(inputs, batch_dims):
        if x_bdim is not None:
            if batch_size is None:
                batch_size = x.shape[x_bdim]
            else:
                assert batch_size == x.shape[x_bdim], "batch sizes must match"
    return batch_size


def maybe_expand_bdim_at_front(x, x_bdim, batch_size):
    if isinstance(x, Tensor):
        if x_bdim is None:
            return x.expand(batch_size, *x.shape)
        return x.movedim(x_bdim, 0)
    else:
        return [x] * batch_size


def _batchify_str():
    return """\
function batch_getindex(xs, i)
    ret = []
    for (idx, x) in enumerate(xs)
        ndx = ndims(x)
        idxs::Vector{Any} = [Colon() for _ in 1:ndx]
        idxs[ndx] = i + 1
        push!(ret, getindex(x, idxs...))
    end
    return Tuple(ret)
end
"""


def maybe_dlpack(xs: list, jl):
    """
    Converts a list of PyTorch tensors to Julia arrays/matrices via DLPack.
    If the list is length 1, returns the single element directly.
    If any entries are not tensors, they are returned as is for PythonCall to convert.
    """
    ret = [dlpack_tensor(x, jl) if isinstance(x, Tensor) else x for x in xs]
    return tuple(ret)


def dlpack_tensor(x: Tensor, jl):
    """
    Convert a single PyTorch tensor to a Julia array/matrix via DLPack.
    """
    return jl.from_dlpack(x.contiguous().detach())  # NOTE: need to use contiguous?


def maybe_init_julia():
    """Start julia by importing juliacall, and load base dependencies"""
    from juliacall import Main

    if Main.isdefined(Main, Main.Symbol("_juliafunction_initialized")) and Main._juliafunction_initialized:
        return

    Main.seval("using DLPack, Distributed")
    Main.seval("_juliafunction_initialized = true")

    init_torch_dlpack()

    Main.seval(_batchify_str())
    Main.seval(_tensorize_str())


def init_torch_dlpack():
    from juliacall import Main

    Main.seval("_torch_from_dlpack = nothing")
    Main._torch_from_dlpack = torch.from_dlpack


def include_files(include: list[str], Main):
    for file in include:
        Main.seval(f'include("{file}")')


def make_module(name, dependencies, globals, jl):
    jl.seval(f"module {name} {dependencies}; {globals} end")
    jl.seval(f"import .{name}")
    return getattr(jl, name)


def _tensorize_str():
    return """\
    
unzip(a) = map(x->getfield.(a, x), fieldnames(eltype(a)))

dlpackable(T) = T in keys(DLPack.jltypes_to_dtypes())

function tensorize(V::Vector{T}) where {T}
    x = copy(V)
    return if dlpackable(T) DLPack.share(x, Main._torch_from_dlpack) else x end
end

function tensorize(V::Vector{Array{T,N}}) where {T,N}
    x = stack(V)
    return if dlpackable(T) DLPack.share(x, Main._torch_from_dlpack) else x end
end
"""
