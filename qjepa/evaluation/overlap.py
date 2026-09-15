"""Merge overlapping IMU predictions by their original trajectory indices."""

from __future__ import annotations

import numpy as np


def triangular_weights(length: int) -> np.ndarray:
    index = np.arange(length, dtype=np.float64)
    return 1.0 - np.abs((2.0 * index - (length - 1)) / (length + 1))


class ImuOverlapMerger:
    def __init__(self, trajectory_length: int, channels: int = 6):
        self.weighted = np.zeros((trajectory_length, channels), dtype=np.float64)
        self.weights = np.zeros(trajectory_length, dtype=np.float64)

    def add(self, start: int, prediction: np.ndarray) -> None:
        if prediction.ndim != 2 or prediction.shape[1] != self.weighted.shape[1]:
            raise ValueError("Prediction must be [L,C]")
        end = start + len(prediction)
        if start < 0 or end > len(self.weighted):
            raise IndexError("Prediction window lies outside the trajectory")
        weight = triangular_weights(len(prediction))
        self.weighted[start:end] += prediction * weight[:, None]
        self.weights[start:end] += weight

    def result(self) -> tuple[np.ndarray, np.ndarray]:
        covered = self.weights > 0
        output = np.full_like(self.weighted, np.nan)
        output[covered] = self.weighted[covered] / self.weights[covered, None]
        return output, covered

