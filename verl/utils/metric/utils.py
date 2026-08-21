# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Metrics utils.
"""

import re
from enum import Enum
from typing import Any, Optional, Union

import numpy as np
import torch

# Matches "min"/"max" only when they appear as a standalone `/`- or `_`-delimited token in a
# metric key (e.g. "loss/min", "min_error"), not as a substring of an unrelated token (e.g.
# "minibatch_loss" or "maximum_length").
_TOKEN_SPLIT_RE = re.compile(r"[^0-9a-zA-Z]+")


def reduce_metrics(metrics: dict[str, Union["Metric", list[Any]]]) -> dict[str, Any]:
    """
    Reduces a dictionary of metric lists by computing the mean, max, or min of each list.
    The reduce operation is determined by the key name:
    - If the key contains "max" as a standalone token (e.g. "foo/max", "foo_max"), np.max is used
    - If the key contains "min" as a standalone token (e.g. "foo/min", "foo_min"), np.min is used
    - Otherwise, np.mean is used

    NOTE: raw ``list`` values are assumed to carry equal weight per entry. If the values being
    reduced represent unequal sample/token counts (e.g. one entry per micro-batch with a
    different number of samples each), wrap them in a :class:`Metric` with explicit per-value
    ``weight`` instead of passing a plain list, so a weighted mean is computed.

    Args:
        metrics: A dictionary mapping metric names to lists of metric values, or to ``Metric``
            instances.

    Returns:
        A dictionary with the same keys but with each list replaced by its reduced value.

    Example:
        >>> metrics = {
        ...     "loss": [1.0, 2.0, 3.0],
        ...     "accuracy": [0.8, 0.9, 0.7],
        ...     "max_reward": [5.0, 8.0, 6.0],
        ...     "min_error": [0.1, 0.05, 0.2]
        ... }
        >>> reduce_metrics(metrics)
        {"loss": 2.0, "accuracy": 0.8, "max_reward": 8.0, "min_error": 0.05}
    """
    for key, val in metrics.items():
        if isinstance(val, Metric):
            metrics[key] = val.aggregate()
        else:
            tokens = _TOKEN_SPLIT_RE.split(key.lower())
            if "max" in tokens:
                metrics[key] = np.max(val)
            elif "min" in tokens:
                metrics[key] = np.min(val)
            else:
                metrics[key] = np.mean(val)
    return metrics


class AggregationType(Enum):
    MEAN = "mean"
    SUM = "sum"
    MIN = "min"
    MAX = "max"


NumericType = int, float, torch.Tensor, np.ndarray
Numeric = int | float | torch.Tensor | np.ndarray


class Metric:
    """
    A metric aggregator for collecting and aggregating numeric values.

    This class accumulates numeric values (int, float, or scalar tensors) and computes
    an aggregate statistic based on the specified aggregation type (MEAN, SUM, MIN, or MAX).

    Each value may optionally carry a ``weight`` (e.g. the number of samples or tokens it was
    computed over). When all weights are 1.0 (the default), aggregation behaves exactly as if
    weights were never used. When weights differ, MEAN is computed as a weighted average
    (``sum(value * weight) / sum(weight)``) and SUM is computed as a weighted sum
    (``sum(value * weight)``), so that entries backed by more samples/tokens contribute
    proportionally more to the aggregate.

    Args:
        aggregation: The aggregation method to use. Can be a string ("mean", "sum", "min", "max")
            or an AggregationType enum value.
        value: Optional initial value(s) to add. Can be a single numeric value or a list of values.
        weight: Optional weight(s) for `value`. Must match the shape of `value` (a single weight
            for a single value, or a list of weights matching a list of values). Defaults to 1.0
            per value when not provided.

    Example:
        >>> metric = Metric(aggregation="mean", value=1.0)
        >>> metric.append(2.0)
        >>> metric.append(3.0)
        >>> metric.aggregate()
        2.0

        >>> # weighted mean: value=1.0 backed by 10 samples, value=3.0 backed by 30 samples
        >>> weighted = Metric(aggregation="mean", value=[1.0, 3.0], weight=[10, 30])
        >>> weighted.aggregate()
        2.5
    """

    def __init__(
        self,
        aggregation: str | AggregationType,
        value: Optional[Numeric | list[Numeric]] = None,
        weight: Optional[Numeric | list[Numeric]] = None,
    ) -> None:
        if isinstance(aggregation, str):
            self.aggregation = AggregationType(aggregation)
        else:
            self.aggregation = aggregation
        if not isinstance(self.aggregation, AggregationType):
            raise ValueError(f"Unsupported aggregation type: {aggregation}")
        self.values = []
        self.weights = []
        if value is not None:
            self.append(value, weight=weight)

    def append(self, value: Union[Numeric, "Metric"], weight: Optional[Numeric] = None) -> None:
        if isinstance(value, Metric):
            self.extend(value)
            return
        if isinstance(value, list):
            self.extend(value, weights=weight)
            return
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError("Only scalar tensors can be converted to float")
            value = value.detach().item()
        if not isinstance(value, NumericType):
            raise ValueError(f"Unsupported value type: {type(value)}")
        self.values.append(value)
        self.weights.append(float(weight) if weight is not None else 1.0)

    def extend(self, values: Union["Metric", list[Numeric]], weights: Optional[list[Numeric]] = None) -> None:
        if isinstance(values, Metric):
            if values.aggregation != self.aggregation:
                raise ValueError(f"Aggregation type mismatch: {self.aggregation} != {values.aggregation}")
            weights = values.weights
            values = values.values
        if weights is None:
            weights = [None] * len(values)
        elif isinstance(weights, (int, float)):
            # a single scalar weight applies uniformly to every value being extended
            weights = [weights] * len(values)
        elif len(weights) != len(values):
            raise ValueError(f"weights must have the same length as values: {len(weights)} != {len(values)}")
        for value, weight in zip(values, weights, strict=True):
            self.append(value, weight=weight)

    def aggregate(self) -> float:
        return self._aggregate(self.values, self.aggregation, self.weights)

    @classmethod
    def _aggregate(
        cls,
        values: list[Numeric],
        aggregation: AggregationType,
        weights: Optional[list[Numeric]] = None,
    ) -> float:
        # Equal (or missing) weights are handled by the plain unweighted reduction so behavior is
        # byte-identical to before per-value weights were introduced.
        is_weighted = weights is not None and not all(w == 1.0 for w in weights)
        match aggregation:
            case AggregationType.MEAN:
                return np.average(values, weights=weights) if is_weighted else np.mean(values)
            case AggregationType.SUM:
                return np.sum(np.multiply(values, weights)) if is_weighted else np.sum(values)
            case AggregationType.MIN:
                return np.min(values)
            case AggregationType.MAX:
                return np.max(values)

    @classmethod
    def aggregate_dp(cls, metric_lists: list["Metric"], weights: Optional[list[Numeric]] = None) -> float:
        """Combines the same metric collected from multiple data-parallel (DP) ranks.

        Args:
            metric_lists: One `Metric` per DP rank. Every rank must hold the same number of
                values (e.g. one value per gradient-accumulation micro-batch) and share the same
                aggregation type.
            weights: Optional per-rank weight (e.g. that rank's local sample or token count), one
                entry per element of `metric_lists`. When provided, MEAN/SUM are combined across
                ranks via a weighted average (`sum(w * v) / sum(w)`) instead of a plain mean, so
                ranks with unequal sample counts are not over/under-counted. Defaults to an
                unweighted mean across ranks, identical to passing equal weights.
        """
        if not metric_lists:
            raise ValueError("Cannot aggregate an empty list of metrics.")
        value_lists = [ml.values for ml in metric_lists]
        if not all(len(ls) == len(value_lists[0]) for ls in value_lists):
            raise ValueError(
                f"All Metric instances must have the same number of values "
                f"for dp aggregation: {[len(ls) for ls in value_lists]}"
            )
        if weights is not None and len(weights) != len(metric_lists):
            raise ValueError(f"weights must have one entry per dp rank: {len(weights)} != {len(metric_lists)}")
        value_arrays = np.array(value_lists)  # [num_dp, num_grad_accumulation]
        aggregation = metric_lists[0].aggregation
        match aggregation:
            case AggregationType.SUM | AggregationType.MEAN:
                # weighted (or, if weights is None, plain) mean over dp ranks
                if weights is not None:
                    combined = np.average(value_arrays, axis=0, weights=weights)
                else:
                    combined = np.mean(value_arrays, axis=0)
                return cls._aggregate(values=combined, aggregation=aggregation)
            case AggregationType.MIN | AggregationType.MAX:
                return cls._aggregate(values=value_arrays.flatten(), aggregation=aggregation)  # min/max over all values

    @classmethod
    def from_dict(cls, data: dict[str, Numeric], aggregation: str | AggregationType) -> dict[str, "Metric"]:
        return {key: cls(value=value, aggregation=aggregation) for key, value in data.items()}

    def init_list(self) -> "Metric":
        return Metric(aggregation=self.aggregation)
