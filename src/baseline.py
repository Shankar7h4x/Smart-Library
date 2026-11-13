"""Serializable baseline estimator used by the occupancy forecasting pipeline."""
import numpy as np


class CalendarMeanBaseline:
    """Weekday/hour mean estimator with a global mean fallback."""
    def __init__(self, grouped_means: dict, overall_mean: float):
        self.grouped_means = dict(grouped_means)
        self.overall_mean = float(overall_mean)

    def predict(self, features):
        return np.asarray([
            self.grouped_means.get((int(row.dayofweek), int(row.hour)), self.overall_mean)
            for row in features.itertuples(index=False)
        ], dtype=float)
