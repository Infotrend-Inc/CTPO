# Adapted from https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
import torch
import math
import datetime

def test_torch(device):
    print(f"On {device}:")

    start = datetime.datetime.now()

    torch.set_default_device(device)

    dtype = torch.float32

    # Create Tensors to hold input and outputs.
    # By default, requires_grad=False, which indicates that we do not need to
    # compute gradients with respect to these Tensors during the backward pass.
    x = torch.linspace(-math.pi, math.pi, 2000, dtype=dtype)
    y = torch.sin(x)

    # Create random Tensors for weights. For a third order polynomial, we need
    # 4 weights: y = a + b x + c x^2 + d x^3
    # Setting requires_grad=True indicates that we want to compute gradients with
    # respect to these Tensors during the backward pass.
    a = torch.randn((), dtype=dtype, requires_grad=True)
    b = torch.randn((), dtype=dtype, requires_grad=True)
    c = torch.randn((), dtype=dtype, requires_grad=True)
    d = torch.randn((), dtype=dtype, requires_grad=True)

    learning_rate = 1e-6
    for t in range(20000):
        # Forward pass: compute predicted y using operations on Tensors.
        y_pred = a + b * x + c * x ** 2 + d * x ** 3

        # Compute and print loss using operations on Tensors.
        # Now loss is a Tensor of shape (1,)
        # loss.item() gets the scalar value held in the loss.
        loss = (y_pred - y).pow(2).sum()
        if t % 1000 == 999:
            print(t, loss.item())

        # Use autograd to compute the backward pass. This call will compute the
        # gradient of loss with respect to all Tensors with requires_grad=True.
        # After this call a.grad, b.grad. c.grad and d.grad will be Tensors holding
        # the gradient of the loss with respect to a, b, c, d respectively.
        loss.backward()

        # Manually update weights using gradient descent. Wrap in torch.no_grad()
        # because weights have requires_grad=True, but we don't need to track this
        # in autograd.
        with torch.no_grad():
            a -= learning_rate * a.grad
            b -= learning_rate * b.grad
            c -= learning_rate * c.grad
            d -= learning_rate * d.grad

            # Manually zero the gradients after updating weights
            a.grad = None
            b.grad = None
            c.grad = None
            d.grad = None

    print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')
    end = datetime.datetime.now()
    print(f"Time taken: {end - start}")
    print("\n\n")

have_gpu = False
if torch.cuda.is_available():
    have_gpu = True

if have_gpu:
    print("Tensorflow test: GPU found")
else:
    print("Tensorflow test: CPU only")

test_torch("cpu")
if have_gpu:
    test_torch("cuda")
# It is possible that the GPU time is slower than the CPU time. This is because the tensor is too small to be worth the overhead of copying to the GPU.
