import os, time

cores = os.cpu_count()

# os.environ["PYTHON_JULIACALL_THREADS"] = str(cores)
# os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"



from juliafunction import ZygoteFunction
import torch

start = time.time()
x = torch.randn(1, 2, 2, requires_grad=True)
f = ZygoteFunction("matrix_relu", setup_code='include("matrix_relu.jl")', batch_dims=(0,))
y = f(x.clone())
y[0].sum().backward()
x.grad
end = time.time()
print(f"Cores: {cores}, Precompilation Time: {end - start:.3f}s")


start = time.time()
x = torch.randn(cores*50, 50, 50, requires_grad=True)
f = ZygoteFunction("matrix_relu", setup_code='include("matrix_relu.jl")', batch_dims=(0,))
y = f(x.clone())
y[0].sum().backward()
x.grad
end = time.time()
print(f"Cores: {cores}, Time: {end - start:.3f}s")