# https://github.com/jump-dev/DiffOpt.jl/blob/821e14e6060d467da680337f9879d70f0b104420/docs/src/examples/custom-relu.jl

using JuMP
import DiffOpt
import Ipopt
import ChainRulesCore
import Base.Iterators: repeated
using LinearAlgebra


function matrix_relu( # 1 -> 3
    y::Matrix;
    model = Model(() -> DiffOpt.diff_optimizer(Ipopt.Optimizer)),
)
    layer_size, batch_size = size(y)
    empty!(model)
    set_silent(model)
    @variable(model, x[1:layer_size, 1:batch_size] >= 0)
    @objective(model, Min, x[:]'x[:] - 2y[:]'x[:])
    optimize!(model)
    return value.(x)
end


function ChainRulesCore.rrule(::typeof(matrix_relu), y::Matrix{T}) where {T}
    model = Model(() -> DiffOpt.diff_optimizer(Ipopt.Optimizer))
    pv = matrix_relu(y; model = model)
    function pullback_matrix_relu(dl_dx)
        # some value from the backpropagation (e.g., loss) is denoted by `l`
        # so `dl_dy` is the derivative of `l` wrt `y`
        x = model[:x] # load decision variable `x` into scope
        dl_dy = zeros(T, size(dl_dx))
        dl_dq = zeros(T, size(dl_dx))
        # set sensitivities
        MOI.set.(model, DiffOpt.ReverseVariablePrimal(), x[:], dl_dx[:])
        # compute grad
        DiffOpt.reverse_differentiate!(model)
        # return gradient wrt objective function parameters
        obj_exp = MOI.get(model, DiffOpt.ReverseObjectiveFunction())
        # coeff of `x` in q'x = -2y'x
        dl_dq[:] .= JuMP.coefficient.(obj_exp, x[:])
        dq_dy = -2 # dq/dy = -2
        dl_dy[:] .= dl_dq[:] * dq_dy
        return (ChainRulesCore.NoTangent(), dl_dy)
    end
    return pv, pullback_matrix_relu
end
