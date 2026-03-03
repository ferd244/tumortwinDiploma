import torch
import torch.nn as nn
from datetime import datetime, timedelta
from typing import Optional, List, Union, Dict

from tumortwin.models.base import TumorGrowthModel3D
from tumortwin.types import RadiotherapySpecification, ChemotherapySpecification, TreatmentTime
from tumortwin.treatments import compute_radiotherapy_cell_survival_fraction, compute_total_cell_death_chemo


class ImmuneTumorModel(TumorGrowthModel3D):
    """
    Модель взаимодействия иммунной системы и опухоли (0D).

    Система ОДУ:
        dx/dt = -λ₁x + α₁ * (x * y^(2/3) * (1 - x/x_c)) / (x + 1)
        dy/dt = λ₂y - α₂ * (x * y^(2/3)) / (x + 1)

    Параметры:
        lambda_1: естественная убыль лимфоцитов
        lambda_2: естественный рост опухоли
        alpha_1: стимуляция лимфоцитов
        alpha_2: уничтожение опухоли
        x_c: предельное количество лимфоцитов
    """

    def __init__(
        self,
        lambda_1: Union[float, torch.Tensor],
        lambda_2: Union[float, torch.Tensor],
        alpha_1: Union[float, torch.Tensor],
        alpha_2: Union[float, torch.Tensor],
        x_c: Union[float, torch.Tensor],
        initial_lymphocytes: Union[float, torch.Tensor],
        initial_tumor: Union[float, torch.Tensor],
        initial_time: Optional[datetime] = None,
        *,
        radiotherapy_specification: Optional[RadiotherapySpecification] = None,
        chemotherapy_specifications: Optional[List[ChemotherapySpecification]] = None,
        require_grad: bool = True,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.device = device

        self.lambda_1 = nn.Parameter(torch.tensor(lambda_1, dtype=torch.float32, device=device), requires_grad=require_grad)
        self.lambda_2 = nn.Parameter(torch.tensor(lambda_2, dtype=torch.float32, device=device), requires_grad=require_grad)
        self.alpha_1 = nn.Parameter(torch.tensor(alpha_1, dtype=torch.float32, device=device), requires_grad=require_grad)
        self.alpha_2 = nn.Parameter(torch.tensor(alpha_2, dtype=torch.float32, device=device), requires_grad=require_grad)
        self.x_c = nn.Parameter(torch.tensor(x_c, dtype=torch.float32, device=device), requires_grad=require_grad)

        self.register_buffer('x', torch.tensor(initial_lymphocytes, dtype=torch.float32, device=device))
        self.register_buffer('y', torch.tensor(initial_tumor, dtype=torch.float32, device=device))

        self.t_initial = initial_time
        self.t_current = initial_time

        self.radiotherapy_specification = radiotherapy_specification
        self.chemotherapy_specifications = chemotherapy_specifications or []
        if radiotherapy_specification and initial_time:
            self.radiotherapy_days = {
                float((day - initial_time).days): dose
                for day, dose in radiotherapy_specification.protocol.items()
            }
        else:
            self.radiotherapy_days = {}

    def forward(self, t: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет производные состояния.

        Args:
            t: текущее время (скалярный тензор)
            u: состояние [x, y] (тензор формы (2,) или (batch, 2))

        Returns:
            du_dt: [dx/dt, dy/dt] той же формы, что и u
        """
        if u.dim() == 1:
            x, y = u[0], u[1]
        else:
            x, y = u[:, 0], u[:, 1]

        x = torch.clamp(x, min=0.0)
        y = torch.clamp(y, min=0.0)

        y_pow = torch.pow(y + 1e-10, 2.0/3.0)

        # dx/dt
        decay = -self.lambda_1 * x
        stimulation = self.alpha_1 * (x * y_pow * (1.0 - x / self.x_c)) / (x + 1.0 + 1e-10)
        dx_dt = decay + stimulation

        # dy/dt
        growth = self.lambda_2 * y
        killing = self.alpha_2 * (x * y_pow) / (x + 1.0 + 1e-10)
        dy_dt = growth - killing

        if u.dim() == 1:
            return torch.stack([dx_dt, dy_dt])
        else:
            return torch.stack([dx_dt, dy_dt], dim=1)

    def step(self, dt: float, method: str = 'euler') -> None:
        """
        Выполняет один шаг интегрирования с учётом лечения.
        Обновляет внутреннее состояние (self.x, self.y, self.t_current).
        """
        if self.t_current is None:
            t_tensor = torch.tensor(0.0, device=self.device)
        else:
            t_tensor = torch.tensor((self.t_current - self.t_initial).total_seconds() / 86400.0, device=self.device)

        # Текущее состояние
        u = torch.stack([self.x, self.y])

        # Интегрирование
        if method == 'euler':
            du = self.forward(t_tensor, u)
            u_new = u + du * dt
        elif method == 'rk4':
            k1 = self.forward(t_tensor, u)
            k2 = self.forward(t_tensor + dt/2, u + k1 * dt/2)
            k3 = self.forward(t_tensor + dt/2, u + k2 * dt/2)
            k4 = self.forward(t_tensor + dt, u + k3 * dt)
            u_new = u + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Применяем лечение (дискретные события)
        x_new, y_new = u_new[0], u_new[1]
        x_new, y_new = self._apply_treatments(t_tensor + dt, x_new, y_new)

        # Ограничения
        x_new = torch.clamp(x_new, min=0.0, max=self.x_c)
        y_new = torch.clamp(y_new, min=0.0)

        # Обновляем состояние
        self.x = x_new
        self.y = y_new
        if self.t_current is not None:
            self.t_current += timedelta(days=dt)

    def _apply_treatments(self, t: torch.Tensor, x: torch.Tensor, y: torch.Tensor):
        """Применяет эффекты радио- и химиотерапии."""
        t_float = float(t)
        # Радиотерапия
        if t_float in self.radiotherapy_days:
            sf = compute_radiotherapy_cell_survival_fraction(
                self.radiotherapy_specification,
                self.radiotherapy_days[t_float]
            )
            y = y * sf
            x = x * sf  # предполагаем одинаковую чувствительность

        # Химиотерапия
        if self.chemotherapy_specifications and self.t_current is not None:
            chemo_effect = compute_total_cell_death_chemo(self.t_current, self.chemotherapy_specifications)
            if chemo_effect > 0:
                # Допустим, химиотерапия убивает часть клеток
                # Здесь нужно использовать соответствующую модель
                # Пока применим как дополнительную смертность
                y = y * (1 - 0.1 * chemo_effect)  # упрощённо
                x = x * (1 - 0.05 * chemo_effect)
        return x, y

    def simulate(self, t_span: tuple, dt: float, method: str = 'rk4') -> Dict:
        """
        Симуляция на интервале времени.
        t_span: (start_time, end_time) объекты datetime
        возвращает историю.
        """
        start, end = t_span
        self.t_initial = start
        self.t_current = start
        self.x = self.x.clone().detach()
        self.y = self.y.clone().detach()

        history = {
            'time': [start],
            'lymphocytes': [float(self.x)],
            'tumor': [float(self.y)]
        }

        steps = int((end - start).total_seconds() / (dt * 86400))
        for _ in range(steps):
            self.step(dt, method)
            history['time'].append(self.t_current)
            history['lymphocytes'].append(float(self.x))
            history['tumor'].append(float(self.y))

        return history
