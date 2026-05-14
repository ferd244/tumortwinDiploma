"""
Проверка, что ``LMoptimizer`` реально меняет ``best_x`` на задаче с известным решением.

Запуск: ``pytest tests/optimizers/test_lm_optimizer.py -v``
Или: ``python tests/optimizers/test_lm_optimizer.py``
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout

import pytest
import torch

from tumortwin.optimizers.lm_optimizer import LMoptimizer, LMoptions


def _silent_steps(optim: LMoptimizer, n: int) -> None:
    """Убираем ``print`` из ``step()`` для читаемого вывода pytest."""
    buf = io.StringIO()
    for _ in range(n):
        with redirect_stdout(buf):
            optim.step()


def test_lm_best_x_moves_on_affine_model():
    """
    y = M @ x,  M фиксирована; цель — восстановить x_true.
    Если best_x остаётся равным начальному приекту, тест падает.
    """
    torch.manual_seed(0)
    M = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [0.5, -1.0]],
        dtype=torch.float64,
    )
    x_true = torch.tensor([0.7, -0.3], dtype=torch.float64)
    y_data = (M @ x_true).reshape(-1)

    def model(x: torch.Tensor) -> torch.Tensor:
        x64 = x.to(dtype=torch.float64)
        return (M @ x64).reshape(-1)

    initial = torch.tensor([0.0, 0.0], dtype=torch.float32)
    bounds = torch.tensor(
        [[-2.0, 2.0], [-2.0, 2.0]],
        dtype=torch.float64,
    )
    optim = LMoptimizer(
        model=model,
        bounds=bounds,
        initial_guess=initial,
        y_data=y_data,
        options=LMoptions(
            jac_delta=1e-5,
            lambda_init=1e-2,
            max_initial_delta=1.0,
        ),
    )
    x0 = optim.best_x.clone()
    _silent_steps(optim, 25)
    final = optim.best_x

    assert not torch.allclose(final, x0, rtol=0, atol=1e-9), (
        "LM не сдвинул best_x; проверьте, что model(x) зависит от x и y_data заданы в том же виде."
    )
    assert torch.allclose(final, x_true, rtol=1e-3, atol=1e-3)
    assert optim.error[-1] < optim.error[0]


def test_lm_best_dtype_is_float64_matching_bounds():
    """Внутренний вектор параметров приводится к float64 (как в оптимизаторе)."""
    M = torch.ones((2, 2), dtype=torch.float64)
    y = torch.tensor([3.0, 3.0], dtype=torch.float64)

    def model(x: torch.Tensor) -> torch.Tensor:
        return (M @ x.to(dtype=torch.float64)).reshape(-1)

    optim = LMoptimizer(
        model=model,
        bounds=torch.tensor([[0.0, 10.0], [0.0, 10.0]], dtype=torch.float64),
        initial_guess=torch.tensor([0.0, 0.0], dtype=torch.float32),
        y_data=y,
        options=LMoptions(lambda_init=0.1, max_initial_delta=1.0),
    )
    assert optim.best_x.dtype == torch.float64
    _silent_steps(optim, 5)
    assert optim.best_x.dtype == torch.float64


if __name__ == "__main__":
    # Быстрая ручная проверка без pytest
    torch.manual_seed(0)
    M = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        dtype=torch.float64,
    )
    x_true = torch.tensor([1.0, 2.0], dtype=torch.float64)
    target = (M @ x_true).reshape(-1)

    def demo_model(x: torch.Tensor) -> torch.Tensor:
        return (M @ x.to(dtype=torch.float64)).reshape(-1)

    initial_guess = torch.tensor([-1.0, 0.5])  # далеко от x_true

    optim = LMoptimizer(
        model=demo_model,
        bounds=torch.tensor([[-5.0, 5.0], [-5.0, 5.0]]),
        initial_guess=initial_guess,
        y_data=target,
        options=LMoptions(jac_delta=1e-5, lambda_init=1e-2, max_initial_delta=1.0),
    )
    print("start best_x:", optim.best_x.tolist())
    print("start error:", float(optim.get_error(optim.model(optim.best_x).reshape(-1), target)))
    for k in range(20):
        optim.step()
        if k % 5 == 4:
            print(f"  iter {k + 1}: best_x={optim.best_x.tolist()}  err={optim.error[-1]:.6g}")
    print("final best_x:  ", optim.best_x.tolist())
    print("true x:        ", x_true.tolist())
    print("error_record (последние 5):", optim.error_record[-5:])
    ok = torch.allclose(optim.best_x, x_true, rtol=1e-2, atol=1e-2)
    print("close to true:", ok)
    sys.exit(0 if ok else 1)
