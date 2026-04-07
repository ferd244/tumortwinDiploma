import torch
import torch.nn as nn
import tqdm.auto as tqdm
from datetime import datetime, timedelta
from typing import ClassVar, List, Optional, Union

from tumortwin.models.pde_system import PDEStateLayout, PDESystemModel3D
from tumortwin.preprocessing import bound_condition_maker
from tumortwin.spatial import FiniteDifferenceOperator3D
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


class ImmuneResponse3D(PDESystemModel3D):
    """
    Пространственная модель иммунного ответа на опухоль.

    Уравнения:
        ∂u1/∂t = D1 ∇²u1 + μ1 u1 (1 - u1 / θ1) - γ12 u1 u4
        ∂u4/∂t = D4 ∇²u4 - v · ∇u4 - γ21 u1 u4 + S(x) (u4^0 - u4)

    где:
        u1 – плотность опухолевых клеток,
        u4 – плотность лимфоцитов,
        D1, D4 – коэффициенты диффузии,
        μ1 – скорость пролиферации опухоли,
        θ1 – локальная предельная плотность опухолевых клеток (логистическое насыщение),
        γ12, γ21 – скорости уничтожения при контакте,
        v – вектор скорости направленного движения лимфоцитов (конвекция),
        S(x) – маска области поступления лимфоцитов (например, кровеносные сосуды),
        u4^0 – концентрация лимфоцитов в крови (источник).

    Лечение (радиотерапия, химиотерапия) добавляется дополнительными членами гибели.
    """

    layout: ClassVar[PDEStateLayout] = PDEStateLayout(num_components=2)

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
        theta1: torch.Tensor = torch.tensor(1.0),     # предельная плотность опухоли
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
        self.theta1 = nn.Parameter(theta1.to(device), requires_grad=require_grad)
        self.gamma12 = nn.Parameter(gamma12.to(device), requires_grad=require_grad)
        self.D4 = nn.Parameter(D4.to(device), requires_grad=require_grad)
        self.gamma21 = nn.Parameter(gamma21.to(device), requires_grad=require_grad)

        # Вектор скорости конвекции
        if isinstance(v, (list, tuple)):
            v = torch.tensor(v, dtype=torch.float32, device=device)
        self.v = nn.Parameter(v.to(device), requires_grad=require_grad)

        u4_src = float(u4_source)
        # Начальные поля
        self.register_buffer("u1_initial", initial_u1.to(device))
        if initial_u4 is None:
            self.register_buffer(
                "u4_initial", torch.full_like(initial_u1.to(device), u4_src)
            )
        else:
            self.register_buffer("u4_initial", initial_u4.to(device))

        # Параметры источника лимфоцитов (как k/d в ReactionDiffusion3D — в оптимизаторе)
        self.u4_source = nn.Parameter(
            torch.tensor(u4_src, dtype=torch.float32, device=device),
            requires_grad=require_grad,
        )
        self.source_rate = nn.Parameter(
            torch.tensor(float(source_rate), dtype=torch.float32, device=device),
            requires_grad=require_grad,
        )
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

        # Предвычисление коэффициентов для конечных разностей (как в ReactionDiffusion3D)
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

        self.chemotherapy_specifications = chemotherapy_specifications
        self.ct_sens = nn.ParameterList(
            [spec.sensitivity for spec in self.chemotherapy_specifications or []]
        )
        self.chemo_sensitivity_tumor = nn.Parameter(
            torch.tensor(float(chemo_sensitivity_tumor), dtype=torch.float32, device=device),
            requires_grad=require_grad,
        )
        self.chemo_sensitivity_lymph = nn.Parameter(
            torch.tensor(float(chemo_sensitivity_lymph), dtype=torch.float32, device=device),
            requires_grad=require_grad,
        )

        # Временные параметры
        self.t_initial = initial_time
        self.progress_bar: Optional[tqdm.tqdm] = None

    def _prepare_fd_stencils(self) -> None:
        """Создаёт ``spatial_fd`` по граничным тегам и шагам сетки (ось 0,1,2 = D,H,W)."""
        sp = self.spacing
        spacing_xyz = [sp.x, sp.y, sp.z]
        self.spatial_fd = FiniteDifferenceOperator3D(self.bcs, spacing_xyz)

    def get_initial_state(self) -> torch.Tensor:
        """
        Специальный метод для подготовки начального тензора для ForwardSolver.
        Объединяет начальные поля опухоли и лимфоцитов в один 4D тензор.
        """
        return torch.stack([self.u1_initial, self.u4_initial], dim=0)

    @torch.enable_grad()
    def forward(self, t: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Вычисление производных по времени для полей u1 и u4.

        Args:
            t: текущее время (дни от начала)
            u: тензор состояния ``(2, D, H, W)`` или ``(batch, 2, D, H, W)``

        Returns:
            du_dt: тензор той же формы
        """
        if self.chemotherapy_specifications:
            if self.t_initial is None:
                raise ValueError(
                    "Unable to compute chemotherapy effect. No initial time set!"
                )

        chemotherapy_effect = (
            compute_total_cell_death_chemo(
                self.t_initial + timedelta(days=t.item()),
                self.chemotherapy_specifications,
            )
            if self.chemotherapy_specifications
            else None
        )

        device = self.device
        u = u.to(device)
        D1 = self.D1.to(device)
        mu1 = self.mu1.to(device)
        theta1 = torch.clamp(self.theta1.to(device), min=1e-6)
        gamma12 = self.gamma12.to(device)
        D4 = self.D4.to(device)
        gamma21 = self.gamma21.to(device)
        v = self.v.to(device)

        # Разделение полей
        if u.dim() == 4:  # (2, D, H, W)
            u1, u4 = u[0], u[1]
        else:  # (batch, 2, D, H, W)
            u1, u4 = u[:, 0], u[:, 1]

        # Применяем ограничения для численной устойчивости
        u1 = torch.clamp(u1, min=0.0)
        u4 = torch.clamp(u4, min=0.0)

        lap1 = self.spatial_fd.laplacian(u1)
        lap4 = self.spatial_fd.laplacian(u4)

        grad_u4 = self.spatial_fd.gradient(u4)
        vb = v.view(3, *([1] * (grad_u4.dim() - 1)))
        convection = -torch.sum(vb * grad_u4, dim=0)

        # Взаимодействие
        interaction12 = gamma12 * u1 * u4
        interaction21 = gamma21 * u1 * u4

        # Источник лимфоцитов (поступление из сосудов)
        source = 0.0
        if self.source_mask is not None:
            sm = self.source_mask.to(device)
            source = self.source_rate * sm * (self.u4_source - u4)

        # Эффект химиотерапии (непрерывный)
        chemo_tumor = 0.0
        chemo_lymph = 0.0
        if chemotherapy_effect is not None:
            chemo_tumor = self.chemo_sensitivity_tumor * chemotherapy_effect
            chemo_lymph = self.chemo_sensitivity_lymph * chemotherapy_effect

        # Производные: логистическая пролиферация предотвращает нереалистичный экспоненциальный рост u1
        growth = mu1 * u1 * (1.0 - torch.clamp(u1, min=0.0) / theta1)
        du1_dt = D1 * lap1 + growth - interaction12 - chemo_tumor * u1
        du4_dt = D4 * lap4 + convection - interaction21 + source - chemo_lymph * u4

        # Сборка результата
        if u.dim() == 4:
            return torch.stack([du1_dt, du4_dt])
        else:
            return torch.stack([du1_dt, du4_dt], dim=1)

    def callback_step(self, t, u, dt):
        """
        Шаг после интегратора (torchdiffeq): радиотерапия, маска, клиппинг.
        """
        if self.progress_bar:
            self.progress_bar.update(dt.item())
        t_float = float(t)
        if (
            self.radiotherapy_specification is not None
            and t_float in self.radiotherapy_days
        ):
            survival = compute_radiotherapy_cell_survival_fraction(
                self.radiotherapy_specification, self.radiotherapy_days[t_float]
            )
            u = u * survival

        if u.dim() == 4:
            u1, u4 = u[0], u[1]
        else:
            u1, u4 = u[:, 0], u[:, 1]

        mask = self.comp_mask.to(u.device).bool()
        u1 = u1 * mask
        u4 = u4 * mask

        u1 = torch.clamp(u1, min=0.0, max=1.0)
        u4 = torch.clamp(u4, min=0.0)

        if u.dim() == 4:
            u = torch.stack([u1, u4])
        else:
            u = torch.stack([u1, u4], dim=1)

        return u

    def callback_step_adjoint(self, t, u, dt):
        """
        Adjoint callback (torchdiffeq): ``u`` — кортеж расширенного состояния
        ``(vjp_t, y, adj_y, *vjp_params)``; согласован с ``ReactionDiffusion3D.callback_step_adjoint``.
        """
        if (
            self.radiotherapy_specification is not None
            and float(t) in self.radiotherapy_days
        ):
            survival = compute_radiotherapy_cell_survival_fraction(
                self.radiotherapy_specification,
                self.radiotherapy_days[float(t)],
            )
            u[2].mul_(survival)

        mask = self.comp_mask.to(u[2].device).bool()
        adj_y = u[2]
        if adj_y.dim() == 4:
            adj_y[0].mul_(mask)
            adj_y[1].mul_(mask)
        elif adj_y.dim() == 5:
            adj_y[:, 0].mul_(mask)
            adj_y[:, 1].mul_(mask)

        return u
