import torch
import torch.nn as nn
from datetime import datetime, timedelta
from typing import Optional, List, Union

from tumortwin.models.base import TumorGrowthModel3D
from tumortwin.preprocessing import bound_condition_maker
from tumortwin.treatments import (
    compute_radiotherapy_cell_survival_fraction,
    compute_total_cell_death_chemo,
)
from tumortwin.types import (
    ChemotherapySpecification,
    RadiotherapySpecification,
    HGGPatientData,
    TNBCPatientData,
)
from tumortwin.types.utility import Boundary


class ImmuneResponse3D(TumorGrowthModel3D):
    """
    Пространственная модель иммунного ответа на опухоль.

    Уравнения:
        ∂u1/∂t = D1 ∇²u1 + μ1 u1 - γ12 u1 u4
        ∂u4/∂t = D4 ∇²u4 - v · ∇u4 - γ21 u1 u4 + S(x) (u4^0 - u4)

    где:
        u1 – плотность опухолевых клеток,
        u4 – плотность лимфоцитов,
        D1, D4 – коэффициенты диффузии,
        μ1 – скорость пролиферации опухоли,
        γ12, γ21 – скорости уничтожения при контакте,
        v – вектор скорости направленного движения лимфоцитов (конвекция),
        S(x) – маска области поступления лимфоцитов (например, кровеносные сосуды),
        u4^0 – концентрация лимфоцитов в крови (источник).

    Лечение (радиотерапия, химиотерапия) добавляется дополнительными членами гибели.
    """

    def __init__(
        self,
        D1: torch.Tensor,
        mu1: torch.Tensor,
        gamma12: torch.Tensor,
        D4: torch.Tensor,
        gamma21: torch.Tensor,
        v: Union[torch.Tensor, List[float]],  # вектор скорости (3,)
        patient_data: Union[HGGPatientData, TNBCPatientData],
        initial_time: datetime,
        *,
        initial_u1: torch.Tensor,                     # начальное поле опухоли
        initial_u4: Optional[torch.Tensor] = None,    # начальное поле лимфоцитов
        u4_source: float = 1.0,                        # концентрация лимфоцитов в крови
        source_mask: Optional[torch.Tensor] = None,    # булева маска области поступления
        source_rate: float = 0.1,                       # скорость поступления (1/день)
        radiotherapy_specification: Optional[RadiotherapySpecification] = None,
        chemotherapy_specifications: Optional[List[ChemotherapySpecification]] = None,
        chemo_sensitivity_tumor: float = 0.8,
        chemo_sensitivity_lymph: float = 0.3,
        require_grad: bool = True,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.device = device

        # Параметры модели (обучаемые)
        self.D1 = nn.Parameter(D1.to(device), requires_grad=require_grad)
        self.mu1 = nn.Parameter(mu1.to(device), requires_grad=require_grad)
        self.gamma12 = nn.Parameter(gamma12.to(device), requires_grad=require_grad)
        self.D4 = nn.Parameter(D4.to(device), requires_grad=require_grad)
        self.gamma21 = nn.Parameter(gamma21.to(device), requires_grad=require_grad)

        # Вектор скорости конвекции
        if isinstance(v, (list, tuple)):
            v = torch.tensor(v, dtype=torch.float32, device=device)
        self.v = nn.Parameter(v.to(device), requires_grad=require_grad)

        # Начальные поля
        self.register_buffer('u1_initial', initial_u1.to(device))
        if initial_u4 is None:
            self.register_buffer('u4_initial', torch.full_like(initial_u1, u4_source))
        else:
            self.register_buffer('u4_initial', initial_u4.to(device))

        # Параметры источника лимфоцитов
        self.u4_source = u4_source
        self.source_rate = source_rate
        if source_mask is not None:
            self.register_buffer('source_mask', source_mask.to(device).bool())
        else:
            self.source_mask = None

        # Маска области и граничные условия (изображение пациента)
        mask_image = (
            patient_data.breastmask_image
            if hasattr(patient_data, "breastmask_image")
            else patient_data.brainmask_image
        )
        self.bcs = torch.from_numpy(bound_condition_maker(mask_image).array).to(device)
        self.comp_mask = torch.from_numpy(mask_image.array).to(device)
        self.spacing = mask_image.spacing  # для шагов сетки

        # Предвычисление коэффициентов для конечных разностей
        self._prepare_fd_stencils()

        # Параметры лечения
        self.radiotherapy_specification = radiotherapy_specification
        if radiotherapy_specification and initial_time:
            self.radiotherapy_days = {
                float((day - initial_time).days): dose
                for day, dose in radiotherapy_specification.protocol.items()
            }
        else:
            self.radiotherapy_days = {}

        self.chemotherapy_specifications = chemotherapy_specifications or []
        self.chemo_sensitivity_tumor = chemo_sensitivity_tumor
        self.chemo_sensitivity_lymph = chemo_sensitivity_lymph

        # Временные параметры
        self.t_initial = initial_time

    def _prepare_fd_stencils(self):
        """Вычисление коэффициентов для оператора Лапласа и градиента."""
        spacing = [self.spacing.x, self.spacing.y, self.spacing.z]
        self.fd_stencil_backward = []
        self.fd_stencil_central = []
        self.fd_stencil_forward = []

        for ax in [0, 1, 2]:
            back_mask = self.bcs[:, :, :, ax] == Boundary.BACKWARD.value
            interior_mask = self.bcs[:, :, :, ax] == Boundary.INTERIOR.value
            forward_mask = self.bcs[:, :, :, ax] == Boundary.FORWARD.value

            inv_dx2 = 1.0 / (spacing[ax] * spacing[ax])
            self.fd_stencil_backward.append(
                self._central_slice(back_mask, ax) * 2.0 * inv_dx2
            )
            self.fd_stencil_central.append(
                self._central_slice(interior_mask, ax) * 1.0 * inv_dx2
            )
            self.fd_stencil_forward.append(
                self._central_slice(forward_mask, ax) * 2.0 * inv_dx2
            )

        # Для конвекции понадобятся веса для первой производной (центральные разности)
        # Можно предвычислить, но проще вычислять на лету с учётом шага.

    @torch.enable_grad()
    def forward(self, t: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Вычисление производных по времени для полей u1 и u4.

        Args:
            t: текущее время (дни от начала)
            u: тензор состояния формы (2, H, W, D) или (batch, 2, H, W, D)

        Returns:
            du_dt: тензор той же формы
        """
        # Разделение полей
        if u.dim() == 4:  # (2, H, W, D)
            u1, u4 = u[0], u[1]
        else:  # (batch, 2, H, W, D)
            u1, u4 = u[:, 0], u[:, 1]

        # Применяем ограничения для численной устойчивости
        u1 = torch.clamp(u1, min=0.0)
        u4 = torch.clamp(u4, min=0.0)

        # Лапласианы
        lap1 = self._compute_laplacian(u1)
        lap4 = self._compute_laplacian(u4)

        # Конвективный член: - v · ∇u4
        # Вычисляем градиент u4 по каждой оси
        grad_u4 = self._compute_gradient(u4)  # список [grad_x, grad_y, grad_z] или тензор (3, H, W, D)
        convection = -torch.sum(self.v[:, None, None, None] * grad_u4, dim=0)  # поэлементное умножение и сумма

        # Взаимодействие
        interaction12 = self.gamma12 * u1 * u4
        interaction21 = self.gamma21 * u1 * u4

        # Источник лимфоцитов (поступление из сосудов)
        source = 0.0
        if self.source_mask is not None:
            source = self.source_rate * self.source_mask * (self.u4_source - u4)

        # Эффект химиотерапии (непрерывный)
        chemo_tumor = 0.0
        chemo_lymph = 0.0
        if self.chemotherapy_specifications and self.t_initial is not None:
            current_time = self.t_initial + timedelta(days=float(t))
            chemo_effect = compute_total_cell_death_chemo(current_time, self.chemotherapy_specifications)
            chemo_tumor = self.chemo_sensitivity_tumor * chemo_effect
            chemo_lymph = self.chemo_sensitivity_lymph * chemo_effect

        # Производные
        du1_dt = self.D1 * lap1 + self.mu1 * u1 - interaction12 - chemo_tumor * u1
        du4_dt = self.D4 * lap4 + convection - interaction21 + source - chemo_lymph * u4

        # Сборка результата
        if u.dim() == 4:
            return torch.stack([du1_dt, du4_dt])
        else:
            return torch.stack([du1_dt, du4_dt], dim=1)

    def callback_step(self, t: torch.Tensor, u: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """
        Обновление состояния после шага интегратора.
        Применяет радиотерапию и ограничения.
        """
        # Радиотерапия (дискретное событие)
        t_float = float(t)
        if self.radiotherapy_specification is not None and t_float in self.radiotherapy_days:
            survival = compute_radiotherapy_cell_survival_fraction(
                self.radiotherapy_specification, self.radiotherapy_days[t_float]
            )
            u = u * survival  # применяем одинаково к обоим полям

        # Разделяем поля для обработки
        if u.dim() == 4:
            u1, u4 = u[0], u[1]
        else:
            u1, u4 = u[:, 0], u[:, 1]

        # Обнуление вне маски
        mask = self.comp_mask.bool()
        u1 = u1 * mask
        u4 = u4 * mask

        # Клиппинг
        u1 = torch.clamp(u1, min=0.0, max=1.0)  # предполагаем нормировку плотности опухоли
        u4 = torch.clamp(u4, min=0.0)           # лимфоциты могут быть любыми

        # Сборка обратно
        if u.dim() == 4:
            u = torch.stack([u1, u4])
        else:
            u = torch.stack([u1, u4], dim=1)

        return u

    # ---------- Вспомогательные методы для конечных разностей ----------
    def _compute_laplacian(self, field: torch.Tensor) -> torch.Tensor:
        """Вычисление лапласиана для одного поля."""
        laplacian = torch.zeros_like(field)
        for ax in [0, 1, 2]:
            back_coeff = self.fd_stencil_backward[ax].to(field.device)
            cent_coeff = self.fd_stencil_central[ax].to(field.device)
            forw_coeff = self.fd_stencil_forward[ax].to(field.device)

            back = self._backward_slice(field, ax)
            cent = self._central_slice(field, ax)
            forw = self._forward_slice(field, ax)

            contrib = (
                back_coeff * (back - cent) +
                cent_coeff * (back - 2*cent + forw) +
                forw_coeff * (forw - cent)
            )
            self._central_slice(laplacian, ax).add_(contrib)
        return laplacian

    def _compute_gradient(self, field: torch.Tensor) -> torch.Tensor:
        """
        Вычисление градиента поля по всем осям.
        Возвращает тензор формы (3, H, W, D) или (3, ...).
        Используются центральные разности, на границах – односторонние.
        """
        grad = torch.zeros((3,) + field.shape, device=field.device, dtype=field.dtype)
        for ax in [0, 1, 2]:
            dx = [self.spacing.x, self.spacing.y, self.spacing.z][ax]
            # Маски границ
            back_mask = self.bcs[:, :, :, ax] == Boundary.BACKWARD.value
            interior_mask = self.bcs[:, :, :, ax] == Boundary.INTERIOR.value
            forward_mask = self.bcs[:, :, :, ax] == Boundary.FORWARD.value

            # Центральная разность для внутренних точек
            if interior_mask.any():
                cent = self._central_slice(field, ax)
                back = self._backward_slice(field, ax)
                forw = self._forward_slice(field, ax)
                grad_cent = (forw - back) / (2 * dx)
                # Запись в центральный срез градиента
                grad_slice = self._central_slice(grad[ax], ax)
                mask_slice = self._central_slice(interior_mask, ax)
                grad_slice[mask_slice] = grad_cent[mask_slice]

            # BACKWARD (нет точки слева) -> forward разность
            if back_mask.any():
                cent = self._central_slice(field, ax)
                forw = self._forward_slice(field, ax)
                grad_back = (forw - cent) / dx
                grad_slice = self._central_slice(grad[ax], ax)
                mask_slice = self._central_slice(back_mask, ax)
                grad_slice[mask_slice] = grad_back[mask_slice]

            # FORWARD (нет точки справа) -> backward разность
            if forward_mask.any():
                cent = self._central_slice(field, ax)
                back = self._backward_slice(field, ax)
                grad_forw = (cent - back) / dx
                grad_slice = self._central_slice(grad[ax], ax)
                mask_slice = self._central_slice(forward_mask, ax)
                grad_slice[mask_slice] = grad_forw[mask_slice]

        return grad

    # Методы для извлечения срезов (как в ReactionDiffusion3D)
    def _backward_slice(self, x: torch.Tensor, ax: int):
        return torch.narrow(x, ax, 0, x.shape[ax] - 2)

    def _central_slice(self, x: torch.Tensor, ax: int):
        return torch.narrow(x, ax, 1, x.shape[ax] - 2)

    def _forward_slice(self, x: torch.Tensor, ax: int):
        return torch.narrow(x, ax, 2, x.shape[ax] - 2)

    def reset(self):
        """Сброс состояния к начальным условиям."""
        self.u1_initial = self.u1_initial.clone()
        self.u4_initial = self.u4_initial.clone()
