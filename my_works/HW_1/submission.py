"""Реализация слоёв GCN и GraphSAGE руками (message passing через scatter).

Контракт соответствует ``GCNConv`` и ``SAGEConv`` из torch_geometric:
при одинаковых весах выход совпадает с PyG с точностью < 1e-5 (см. автотесты
``tasks/tests/test_task1.py``). Агрегация — через ``torch_geometric.utils.scatter``.

MyGCNConv (Kipf & Welling, 2017):
    h_v = W^T * ( sum_u x_u / sqrt(d_v * d_u) ),  u in N(v) ∪ {v},
где d_v — степень вершины v ВКЛЮЧАЯ self-петлю.

MySAGEConv (Hamilton et al., 2017, агрегатор mean):
    h_v = W_self^T x_v + W_neigh^T * mean(x_u : u in N(v)).
Собственный вектор идёт через отдельную матрицу, self-петли НЕ добавляются.

ВАЖНО про тесты: эталонный ``GCNConv``/``SAGEConv`` создаётся там с
``bias=False``, поэтому копируется только ``weight``, а bias эталона остаётся
нулевым. Чтобы выход совпадал с таким эталоном, оба слоя здесь по умолчанию
создаются БЕЗ bias (``bias=False``). В ноутбуке для обучения моделей
передайте ``bias=True`` — это ровно то же поведение, что у слоёв PyG.
"""

from typing import Optional

import torch
from torch import Tensor
from torch.nn import Linear
from torch_geometric.utils import scatter, add_remaining_self_loops

__all__ = ["MyGCNConv", "MySAGEConv"]


class MyGCNConv(torch.nn.Module):
    """Слой GCN: нормированная сумма сообщений по целевой вершине + self-петли.

    Параметры
    ---------
    in_channels:
        размерность входных признаков вершин;
    out_channels:
        размерность выходных признаков вершин;
    bias:
        добавлять ли смещение после линейного преобразования. По умолчанию
        выключен — чтобы совпадать с эталонным ``GCNConv(..., bias=False)``
        из автотестов (тесты копируют только ``lin.weight``). Для обучения
        моделей передайте ``bias=True``.

    Атрибуты
    --------
    lin : torch.nn.Linear
        единственная матрица весов (тесты копируют в неё веса эталонного слоя).
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = torch.nn.Parameter(torch.zeros(out_channels)) if bias else None

    def reset_parameters(self) -> None:
        """Переинициализация весов тем же распределением, что в конструкторе."""
        self.lin.reset_parameters()
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Признаки (n, in_channels) + рёбра (2, E) -> выход (n, out_channels).

        Сообщения собираются по ЦЕЛЕВОЙ вершине (второй строке ``edge_index``),
        self-петли добавляются, степени считаются с петлями; у изолированных
        вершин выход конечен (деления на ноль нет по построению).
        """
        num_nodes = x.size(0)
        row, col = edge_index  # источник, цель

        # Self-петли: каждая вершина сначала получает своё собственное сообщение.
        edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=num_nodes)
        src, dst = edge_index

        # Степени с учётом петель: количество входящих сообщений на вершину.
        deg = torch.bincount(dst, minlength=num_nodes).to(x.dtype)

        # Нормировка GCN: 1 / sqrt(d_v * d_u) для каждого ребра (включая петли).
        deg_inv_sqrt = deg.clamp(min=1.0).pow(-0.5)
        edge_weight = deg_inv_sqrt[dst] * deg_inv_sqrt[src]

        # Пропускаем признаки через линейный слой до агрегации
        # (эквивалентно агрегации сырых признаков с последующим W, т.к. сумма
        # линейна — порядок не влияет на результат, как у PyG).
        x = self.lin(x)

        # Размножаем признаки по рёбрам и взвешиваем.
        messages = x.index_select(0, src) * edge_weight.unsqueeze(-1)

        # Сумма по целевой вершине. Вершина без входящих рёбер (кроме петли)
        # получает только собственный вклад x_v / d_v — вывод конечен.
        out = scatter(messages, dst, dim=0, dim_size=num_nodes, reduce="sum")
        return out if self.bias is None else out + self.bias

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.in_channels}, {self.out_channels})"


class MySAGEConv(torch.nn.Module):
    """Слой GraphSAGE с mean-агрегатором и отдельной матрицей для собственного вектора.

    Параметры
    ---------
    in_channels:
        размерность входных признаков вершин;
    out_channels:
        размерность выходных признаков вершин;
    bias:
        добавлять ли смещение. По умолчанию выключен — чтобы совпадать с
        эталонным ``SAGEConv(..., bias=False)`` из автотестов (тесты копируют
        только веса ``lin_l``/``lin_r``). Для обучения моделей передайте
        ``bias=True``.

    Атрибуты
    --------
    lin_neigh : torch.nn.Linear
        матрица для усреднённого вектора соседей (соответствует ``lin_l`` в PyG);
    lin_self : torch.nn.Linear
        матрица для собственного вектора вершины (соответствует ``lin_r`` в PyG).
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin_neigh = Linear(in_channels, out_channels, bias=bias)
        self.lin_self = Linear(in_channels, out_channels, bias=False)

    def reset_parameters(self) -> None:
        """Переинициализация весов обеих матриц."""
        self.lin_neigh.reset_parameters()
        self.lin_self.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Признаки (n, in_channels) + рёбра (2, E) -> выход (n, out_channels).

        Сообщения собираются по целевой вершине (второй строке ``edge_index``).
        Self-петли НЕ добавляются: собственный вектор проходит через отдельную
        матрицу ``lin_self``, а соседний — через ``lin_neigh``.
        """
        num_nodes = x.size(0)
        row, col = edge_index  # источник, цель

        # Степень = число входящих сообщений по целевой вершине.
        deg = torch.bincount(col, minlength=num_nodes).clamp(min=1).to(x.dtype)

        # Сумма соседских сообщений по целевой вершине (для вершины без соседей
        # останется нулевой вектор), затем делим на число соседей.
        messages = x.index_select(0, row)
        neigh_sum = scatter(
            messages, col, dim=0, dim_size=num_nodes, reduce="sum"
        )
        neigh_mean = neigh_sum / deg.unsqueeze(-1)

        # Собственный вектор и усреднённое соседство идут через РАЗНЫЕ матрицы.
        out = self.lin_self(x) + self.lin_neigh(neigh_mean)
        return out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.in_channels}, {self.out_channels})"
