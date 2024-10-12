from __future__ import annotations
from typing import Optional

from torch.nn import Module

from juliafunction.utils import (
    maybe_init_julia,
    include_files,
    maybe_dlpack,
    detect_batch_size,
    maybe_expand_bdim_at_front,
    make_module
)


class JuliaFunction(Module):
    """A PyTorch layer that wraps two Julia functions, one for forward and one for back.
    Assumes the functions are in the form:
    ```
        out, _state = forward(args...)
        grad_args = backward(_state, grad_out...)
    ```

    Args:
        forward (str): The name of the Julia function to call in the forward pass.
        backward (str): The name of the Julia function to call in the backward pass.
        batch_dims (list[int]): The batch dimensions of the inputs, None if broadcasting.
        setup_code (str): Julia code to execute before each forward pass.
        include (str | list[str]): Julia files to include before each forward pass.
    """

    def __init__(
        self,
        forward: str,
        backward: str,
        batch_dims: Optional[list[int]],
        setup_code: str = "",
        include: Optional[str | list[str]] = None,
    ):
        super().__init__()

        maybe_init_julia()

        self._forward = _make_julia_function(
            fwd_str=f"_out = @spawnat :any {forward}(_args...)",
            bwd_str=f"_out = @spawnat :any {backward}(_state, _args)",
            fwd_cleanup_str="_args = nothing; _out = nothing",
            bwd_cleanup_str="_args = nothing; _out = nothing; _state = nothing",
            setup_code=setup_code,
            include=include,
            dependencies="using DLPack, Distributed",
            global_vars="global _args, _state, _out",
            sample_imports=[forward, backward],
            batch_dims=batch_dims,
        )

    def forward(self, *inputs):
        return self._forward(*inputs)


class ZygoteFunction(Module):
    """A PyTorch layer that wraps a Zygote.jl-differentiable function.

    Args:
        fn_name (str): The name of the Julia function to wrap.
        batch_dims (list[int]): The batch dimensions of the inputs, None if broadcasting.
        setup_code (str): Julia code to execute before each forward pass.
        include (str | list[str]): Julia files to include before each forward pass.
    """

    def __init__(
        self,
        fn_name: str,
        batch_dims: Optional[list[int]],
        setup_code: str = "",
        include: Optional[str | list[str]] = None,
    ):
        super().__init__()

        maybe_init_julia()

        self._forward = _make_julia_function(
            fwd_str=f"_out = @spawnat :any Zygote.pullback({fn_name}, _args...)",
            bwd_str=f"_out = @spawnat :any _state(_args)",
            fwd_cleanup_str="_args = nothing; _out = nothing",
            bwd_cleanup_str="_args = nothing; _out = nothing; _state = nothing",
            setup_code=setup_code,
            include=include,
            dependencies="using DLPack, Zygote, Distributed",
            global_vars="global _args, _state, _out",
            sample_imports=[fn_name],
            batch_dims=batch_dims,
        )

    def forward(self, *inputs):
        return self._forward(*inputs)


def _make_julia_function(
    fwd_str: str,
    bwd_str: str,
    fwd_cleanup_str: str = "",
    bwd_cleanup_str: str = "",
    setup_code: str = "",
    include: Optional[str | list[str]] = None,
    dependencies: str = "",
    global_vars: str = "",
    sample_imports: list[str] = None,
    batch_dims: Optional[list[int]] = None,
):
    from uuid import uuid4
    from juliacall import Main
    from torch.autograd import Function
    from torch.autograd.function import once_differentiable

    assert dependencies == "" or dependencies.startswith(
        ("using ", "import ")
    ), "dependencies must be a string of Julia imports"
    assert global_vars == "" or global_vars.startswith("global "), "globals must be a string of global variables"

    if include is None:
        include = []
    elif isinstance(include, str):
        include = [include]

    base_globals = "global _inputs, _batch_dims, _outputs, _errors"

    sample_data = "{_samplemodule}._args = Main.batch_getindex(_inputs, {i})"

    unbatch_single_grady = """
        {_samplemodule}._args =
            if length({_samplemodule}._args) == 1
                {_samplemodule}._args[1]
            else
                {_samplemodule}._args
            end
    """

    fetch_fwd = """
        _outputs, _state = Main.unzip(
            fetch.(
                [s._out for s in [eval(Meta.parse("{_callmodule}_$(i-1)")) for i in 1:{batch_size}]]
            )
        )
    """
    fetch_bwd = """
        _outputs = Main.unzip(fetch.(
            [s._out for s in [eval(Meta.parse("{_callmodule}_$(i-1)")) for i in 1:{batch_size}]]
        ))
    """
    pack = """
        if _outputs[1][1] isa Tuple
            [Main.tensorize(i) for i in Main.unzip(_outputs[1])]
        elseif _outputs[1] isa Tuple
            [Main.tensorize(i) for i in Main.unzip(_outputs)]
        elseif _outputs isa Tuple
            [Main.tensorize(i) for i in _outputs]
        else
            [Main.tensorize(_outputs)]
        end
    """

    class _JuliaFunction(Function):
        @staticmethod
        def setup_context(ctx, _, outputs):
            ctx._callmodule, ctx._batch_size = outputs[0]

        @staticmethod
        def forward(*inputs):
            assert len(inputs) == len(batch_dims), f"batch dims must match inputs. Got {len(inputs)} inputs and {len(batch_dims)} batch dims ({batch_dims})."
            batch_size = detect_batch_size(inputs, batch_dims)

            # make the module for this forward pass
            # each sample will get its own module, inside of this one.
            _callmodule = f"JFMCall_{uuid4().hex}"
            CallModule = make_module(_callmodule, dependencies, base_globals, Main)

            # evaluate setup code
            # NOTE: users may want to manually evaluate their setup code in Main if it is expensive
            include_files(include, CallModule)
            CallModule.seval(setup_code)

            # batchify the non-batched inputs
            expanded_inputs = [maybe_expand_bdim_at_front(x, x_bdim, batch_size) for x, x_bdim in zip(inputs, batch_dims)]

            # send the batched inputs to julia
            CallModule._inputs = maybe_dlpack(expanded_inputs, CallModule)
            CallModule._batch_dims = batch_dims

            # @spawnat loop
            for i in range(batch_size):
                # make the module for this sample
                _samplemodule = f"{_callmodule}_{i}"
                SampleModule = make_module(_samplemodule, dependencies, global_vars, CallModule)

                # import the functions from the call module
                SampleModule.seval(f"import ..{_callmodule}: {', '.join(sample_imports)}")

                # set the data for this sample
                CallModule.seval(sample_data.format(_samplemodule=_samplemodule, i=i))

                # run the forward pass
                SampleModule.seval(fwd_str)

            # fetch the results
            CallModule.seval(fetch_fwd.format(_callmodule=_callmodule, batch_size=batch_size))
            y = CallModule.seval(pack)

            # clean up
            for i in range(batch_size):
                SampleModule = getattr(CallModule, f"{_callmodule}_{i}")
                SampleModule.seval(f"_state = Main.{_callmodule}._state[{i+1}]")
                SampleModule.seval(fwd_cleanup_str)
            CallModule.seval(f"_inputs = nothing; _outputs = nothing")

            return (_callmodule, batch_size), *y

        @staticmethod
        @once_differentiable
        def backward(ctx, _, *grad_ys):
            # get module
            CallModule = getattr(Main, ctx._callmodule)

            # dlpack grad_ys
            CallModule._inputs = maybe_dlpack(grad_ys, CallModule)

            # @spawnat loop
            for i in range(ctx._batch_size):
                # get this sample's module
                _samplemodule = f"{ctx._callmodule}_{i}"
                SampleModule = getattr(CallModule, _samplemodule)

                # set the data for this sample
                CallModule.seval(sample_data.format(_samplemodule=_samplemodule, i=i))

                # unbatch if only one grad_y, for compat with Zygote
                SampleModule.seval(unbatch_single_grady.format(_samplemodule=_samplemodule))

                # run the backward pass
                SampleModule.seval(bwd_str)

            # fetch the results
            CallModule.seval(fetch_bwd.format(_callmodule=ctx._callmodule, batch_size=ctx._batch_size))
            y = CallModule.seval(pack)

            # clean up
            for i in range(ctx._batch_size):
                SampleModule = getattr(CallModule, f"{ctx._callmodule}_{i}")
                SampleModule.seval(bwd_cleanup_str)
            CallModule.seval(f"inputs = nothing; batch_dims = nothing; outputs = nothing; _errors = nothing")

            return tuple(y)

    def julia_function(*args, **kwargs):
        not_first = _JuliaFunction.apply(*args, **kwargs)[1:]
        return not_first if len(not_first) > 1 else not_first[0]

    return julia_function
