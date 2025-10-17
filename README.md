# juliafunction

> [!WARNING]
> This is probably broken in Julia 1.12. Proceed with caution.

Utilities for embedding differentiable Julia functions in PyTorch training pipelines, using JuliaCall.

Usage:
```python
setup_code = """
function add_or_mul(x, y, mode)
  if mode == "add"
    return x .+ y
  else
    return x .* y
  end
end"""

add_or_mul_layer = ZygoteFunction("add_or_mul", batch_dims=(0,0,None), setup_code=setup_code)

x = torch.randn(4, 8, requires_grad=True)
y = torch.randn(4, 8, requires_grad=True)

add_ = add_or_mul_layer(x, y, "add")
mul_ = add_or_mul_layer(x, y, "mul")

(add_ + mul_).mean().backward()
x.grad
```
