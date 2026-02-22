
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

try:
    from statsmodels.tsa.seasonal import STL

    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False


EPS = 1e-8


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    z = x - np.max(x)
    e = np.exp(z)
    return e / (np.sum(e) + EPS)


def ridge_pinv_solution(phi: np.ndarray, y: np.ndarray, reg_lambda: float) -> np.ndarray:
    """Closed-form ridge solution using pseudo-inverse.

    beta = pinv(Phi^T Phi + lambda I) Phi^T y
    """
    n_features = phi.shape[1]
    a = phi.T @ phi + reg_lambda * np.eye(n_features)
    return np.linalg.pinv(a) @ phi.T @ y


def rls_update(
    w: np.ndarray,
    p: np.ndarray,
    phi: np.ndarray,
    target: float,
    forgetting_factor: float = 0.995,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Single-sample Recursive Least Squares update."""
    w_col = w.reshape(-1, 1)
    phi_col = phi.reshape(-1, 1)

    pred = (phi_col.T @ w_col).item()
    err = target - pred

    denom = forgetting_factor + (phi_col.T @ p @ phi_col).item()
    gain = (p @ phi_col) / (denom + EPS)

    w_new = w_col + gain * err
    p_new = (p - gain @ phi_col.T @ p) / max(forgetting_factor, 1e-6)

    return w_new.ravel(), p_new, err


def make_sliding_windows(series: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Convert 1D series into supervised (X, y) with fixed look-back window."""
    if len(series) <= window_size:
        raise ValueError("Series length must be greater than window_size.")

    x, y = [], []
    for i in range(len(series) - window_size):
        x.append(series[i : i + window_size])
        y.append(series[i + window_size])

    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def _fallback_decompose_1d(window: np.ndarray, period: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fallback additive decomposition: trend + seasonal + residual."""
    n = len(window)
    period = int(max(2, min(period, n)))

    kernel_size = max(3, min(n - (1 - n % 2), period if period % 2 == 1 else period - 1))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = min(kernel_size, n if n % 2 == 1 else n - 1)
    kernel_size = max(3, kernel_size)

    kernel = np.ones(kernel_size, dtype=float) / kernel_size
    trend = np.convolve(window, kernel, mode="same")

    detrended = window - trend
    seasonal = np.zeros_like(window)
    for p in range(period):
        idx = np.arange(p, n, period)
        if len(idx) > 0:
            seasonal[idx] = np.mean(detrended[idx])

    residual = window - trend - seasonal
    return trend, seasonal, residual


def decompose_window(
    window: np.ndarray,
    period: int = 24,
    use_stl: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose one window into trend, seasonal, residual components."""
    window = np.asarray(window, dtype=float)

    if use_stl and HAS_STATSMODELS and len(window) >= max(2 * period, 8):
        safe_period = int(max(2, min(period, len(window) // 2)))
        try:
            stl = STL(window, period=safe_period, robust=True)
            result = stl.fit()
            trend = np.asarray(result.trend, dtype=float)
            seasonal = np.asarray(result.seasonal, dtype=float)
            residual = np.asarray(result.resid, dtype=float)
            return trend, seasonal, residual
        except Exception:
            pass

    return _fallback_decompose_1d(window, period=period)


def decompose_batch(
    x_windows: np.ndarray,
    period: int = 24,
    use_stl: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decompose each row window and return concatenated features."""
    n, w = x_windows.shape
    trend_all = np.zeros((n, w), dtype=float)
    seasonal_all = np.zeros((n, w), dtype=float)
    resid_all = np.zeros((n, w), dtype=float)

    for i in range(n):
        trend, seasonal, resid = decompose_window(x_windows[i], period=period, use_stl=use_stl)
        trend_all[i] = trend
        seasonal_all[i] = seasonal
        resid_all[i] = resid

    x_decomp = np.hstack([trend_all, seasonal_all, resid_all])
    return x_decomp, trend_all, seasonal_all, resid_all


def build_gate_features(x_windows: np.ndarray) -> np.ndarray:
    """Feature extractor for adaptive residual gate."""
    mean_ = np.mean(x_windows, axis=1)
    std_ = np.std(x_windows, axis=1)
    var_ = np.var(x_windows, axis=1)
    min_ = np.min(x_windows, axis=1)
    max_ = np.max(x_windows, axis=1)
    rng_ = max_ - min_
    slope_ = (x_windows[:, -1] - x_windows[:, 0]) / (x_windows.shape[1] + EPS)
    diff_std_ = np.std(np.diff(x_windows, axis=1), axis=1)

    return np.column_stack([mean_, std_, var_, rng_, slope_, diff_std_])


def summarize_decomposition_components(
    trend: np.ndarray,
    seasonal: np.ndarray,
    residual: np.ndarray,
) -> Dict[str, str]:
    """Human-readable decomposition labels for CLI report."""
    trend_slope = float(trend[-1] - trend[0])
    trend_scale = float(np.std(trend) + EPS)

    if trend_slope > 0.2 * trend_scale:
        trend_label = "Rising"
    elif trend_slope < -0.2 * trend_scale:
        trend_label = "Falling"
    else:
        trend_label = "Stable"

    seasonal_strength = float(np.std(seasonal) / (np.std(trend) + EPS))
    if seasonal_strength >= 0.9:
        seasonal_label = "High"
    elif seasonal_strength >= 0.4:
        seasonal_label = "Moderate"
    else:
        seasonal_label = "Low"

    resid_strength = float(np.std(residual) / (np.std(trend + seasonal) + EPS))
    if resid_strength >= 0.9:
        resid_label = "High Noise"
    elif resid_strength >= 0.5:
        resid_label = "Medium Noise"
    else:
        resid_label = "Low Noise"

    return {
        "trend": trend_label,
        "seasonal": seasonal_label,
        "residual": resid_label,
    }


class TinyMLPGate:
    """Tiny MLP gate for adaptive residual scaling alpha in [0, 1]."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 8,
        learning_rate: float = 0.01,
        random_state: Optional[int] = None,
    ) -> None:
        rng = np.random.default_rng(random_state)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate

        self.w1 = rng.normal(0, 0.25, size=(input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim, dtype=float)
        self.w2 = rng.normal(0, 0.25, size=(hidden_dim, 1))
        self.b2 = np.zeros(1, dtype=float)

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))

    def _forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        h_raw = x @ self.w1 + self.b1
        h = np.tanh(h_raw)
        z = h @ self.w2 + self.b2
        y = self._sigmoid(z).ravel()
        return h, z, y

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        _, _, y = self._forward(x)
        return np.clip(y, 0.02, 1.0)

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int = 250) -> None:
        self.partial_fit(x, y, epochs=epochs)

    def partial_fit(self, x: np.ndarray, y: np.ndarray, epochs: int = 10) -> None:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        if x.ndim == 1:
            x = x.reshape(1, -1)

        n = x.shape[0]
        lr = self.learning_rate

        for _ in range(epochs):
            h, z, y_hat = self._forward(x)

            # MSE gradient through sigmoid output.
            grad_out = 2.0 * (y_hat - y) / max(n, 1)
            sig = self._sigmoid(z).ravel()
            grad_z = (grad_out * sig * (1.0 - sig)).reshape(-1, 1)

            grad_w2 = h.T @ grad_z
            grad_b2 = np.sum(grad_z, axis=0)

            grad_h = grad_z @ self.w2.T
            grad_h_raw = grad_h * (1.0 - np.square(h))

            grad_w1 = x.T @ grad_h_raw
            grad_b1 = np.sum(grad_h_raw, axis=0)

            self.w2 -= lr * grad_w2
            self.b2 -= lr * grad_b2
            self.w1 -= lr * grad_w1
            self.b1 -= lr * grad_b1


@dataclass
class RVFLLayerState:
    w_hidden: np.ndarray
    b_hidden: np.ndarray
    beta_out: np.ndarray
    p_inv: np.ndarray


class ResdRVFLBlock:
    """One residual deep RVFL block with adaptive residual scaling gate."""

    def __init__(
        self,
        n_layers: int,
        hidden_units: List[int],
        reg_lambda: float,
        forgetting_factor: float,
        gate_input_dim: int,
        gate_hidden_dim: int,
        random_state: Optional[int] = None,
    ) -> None:
        if len(hidden_units) != n_layers:
            raise ValueError("hidden_units length must match n_layers.")

        self.n_layers = n_layers
        self.hidden_units = hidden_units
        self.reg_lambda = reg_lambda
        self.forgetting_factor = forgetting_factor
        self.rng = np.random.default_rng(random_state)

        self.gate = TinyMLPGate(
            input_dim=gate_input_dim,
            hidden_dim=gate_hidden_dim,
            learning_rate=0.01,
            random_state=random_state,
        )

        self.var_reference_ = 1.0
        self.layers_: List[RVFLLayerState] = []
        self.feature_dim_: Optional[int] = None

    def _alpha_targets(self, gate_features: np.ndarray) -> np.ndarray:
        """Supervision target for gate from input variance statistics."""
        var = gate_features[:, 2]
        self.var_reference_ = float(np.median(var) + EPS)
        alpha = var / (2.0 * self.var_reference_ + EPS)
        return np.clip(alpha, 0.05, 1.0)

    def _init_layers(self, feature_dim: int) -> None:
        self.layers_ = []
        self.feature_dim_ = feature_dim

        prev_hidden_dim = 0
        for i in range(self.n_layers):
            in_dim = feature_dim if i == 0 else feature_dim + prev_hidden_dim
            h_dim = self.hidden_units[i]

            w_hidden = self.rng.normal(0, 1.0, size=(in_dim, h_dim))
            b_hidden = self.rng.normal(0, 0.2, size=(h_dim,))

            beta_out = np.zeros(feature_dim + h_dim, dtype=float)
            p_inv = np.eye(feature_dim + h_dim, dtype=float)

            self.layers_.append(
                RVFLLayerState(
                    w_hidden=w_hidden,
                    b_hidden=b_hidden,
                    beta_out=beta_out,
                    p_inv=p_inv,
                )
            )
            prev_hidden_dim = h_dim

    @staticmethod
    def _activation(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def fit(self, x_feat: np.ndarray, y: np.ndarray, gate_features: np.ndarray) -> np.ndarray:
        """Fit block with residual learning and closed-form output weights."""
        n_samples, feature_dim = x_feat.shape
        if n_samples != len(y):
            raise ValueError("x_feat and y size mismatch.")

        if not self.layers_:
            self._init_layers(feature_dim)

        alpha_targets = self._alpha_targets(gate_features)
        self.gate.fit(gate_features, alpha_targets, epochs=250)
        alpha = self.gate.predict(gate_features)

        pred_total = np.zeros(n_samples, dtype=float)
        prev_hidden = None

        for li, layer in enumerate(self.layers_):
            layer_in = x_feat if li == 0 else np.hstack([x_feat, prev_hidden])
            hidden = self._activation(layer_in @ layer.w_hidden + layer.b_hidden)
            phi = np.hstack([x_feat, hidden])

            if li == 0:
                phi_eff = phi
                target = y
            else:
                phi_eff = phi * alpha[:, None]
                target = y - pred_total

            beta = ridge_pinv_solution(phi_eff, target, reg_lambda=self.reg_lambda)
            pred_layer = phi_eff @ beta

            if li == 0:
                pred_total = pred_layer
            else:
                pred_total = pred_total + pred_layer

            p_inv = np.linalg.pinv(phi_eff.T @ phi_eff + self.reg_lambda * np.eye(phi_eff.shape[1]))

            layer.beta_out = beta
            layer.p_inv = p_inv
            prev_hidden = hidden

        return pred_total

    def predict(self, x_feat: np.ndarray, gate_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict block output and alpha gate values."""
        if not self.layers_:
            raise RuntimeError("Block is not fitted.")

        alpha = self.gate.predict(gate_features)
        pred_total = np.zeros(x_feat.shape[0], dtype=float)

        prev_hidden = None
        for li, layer in enumerate(self.layers_):
            layer_in = x_feat if li == 0 else np.hstack([x_feat, prev_hidden])
            hidden = self._activation(layer_in @ layer.w_hidden + layer.b_hidden)
            phi = np.hstack([x_feat, hidden])

            if li == 0:
                pred_layer = phi @ layer.beta_out
                pred_total = pred_layer
            else:
                pred_layer = (phi * alpha[:, None]) @ layer.beta_out
                pred_total = pred_total + pred_layer

            prev_hidden = hidden

        return pred_total, alpha

    def update_one(self, x_feat: np.ndarray, gate_feat: np.ndarray, y_true: float) -> float:
        """Single-sample online update (RLS) and return pre-update prediction."""
        if not self.layers_:
            raise RuntimeError("Block is not fitted.")

        x_feat = np.asarray(x_feat, dtype=float).reshape(1, -1)
        gate_feat = np.asarray(gate_feat, dtype=float).reshape(1, -1)

        # Online gate adaptation using variance-driven target.
        alpha_target = np.clip(gate_feat[0, 2] / (2.0 * self.var_reference_ + EPS), 0.05, 1.0)
        self.gate.partial_fit(gate_feat, np.array([alpha_target], dtype=float), epochs=8)
        alpha = float(self.gate.predict(gate_feat)[0])

        pred_total = 0.0
        prev_hidden = None

        for li, layer in enumerate(self.layers_):
            layer_in = x_feat if li == 0 else np.hstack([x_feat, prev_hidden])
            hidden = self._activation(layer_in @ layer.w_hidden + layer.b_hidden)
            phi = np.hstack([x_feat, hidden]).ravel()

            if li == 0:
                phi_eff = phi
                target = float(y_true)
            else:
                phi_eff = phi * alpha
                target = float(y_true - pred_total)

            pred_layer = float(phi_eff @ layer.beta_out)
            if li == 0:
                pred_total = pred_layer
            else:
                pred_total += pred_layer

            beta_new, p_new, _ = rls_update(
                w=layer.beta_out,
                p=layer.p_inv,
                phi=phi_eff,
                target=target,
                forgetting_factor=self.forgetting_factor,
            )
            layer.beta_out = beta_new
            layer.p_inv = p_new

            prev_hidden = hidden

        return pred_total


class AdaptiveSResdRVFL:
    """Adaptive Attention-based Stacked Residual RVFL for time-series forecasting."""

    def __init__(
        self,
        window_size: int = 50,
        n_blocks: int = 5,
        n_layers: int = 3,
        hidden_units: Optional[List[int]] = None,
        reg_lambda: float = 1e-3,
        period: int = 24,
        use_stl: bool = True,
        attention_temperature: float = 0.08,
        attention_error_power: float = 2.0,
        forgetting_factor: float = 0.995,
        performance_decay: float = 0.85,
        random_state: int = 42,
    ) -> None:
        self.window_size = window_size
        self.n_blocks = n_blocks
        self.n_layers = n_layers
        self.hidden_units = hidden_units if hidden_units is not None else [64] * n_layers
        self.reg_lambda = reg_lambda
        self.period = period
        self.use_stl = use_stl
        self.attention_temperature = attention_temperature
        self.attention_error_power = attention_error_power
        self.forgetting_factor = forgetting_factor
        self.performance_decay = performance_decay
        self.random_state = random_state

        self.scaler = MinMaxScaler()
        self.blocks: List[ResdRVFLBlock] = []
        self.block_mse_ = np.ones(self.n_blocks, dtype=float)
        self.is_fitted_ = False

    def _attention_weights(self) -> np.ndarray:
        # Relative error + nonlinear sharpening => faster separation of good/bad blocks.
        mse = np.clip(self.block_mse_.astype(float), EPS, None)
        relative_mse = mse / (np.mean(mse) + EPS)
        logits = -(relative_mse ** self.attention_error_power) / max(self.attention_temperature, 1e-4)
        return softmax(logits)

    def fit(self, series: np.ndarray) -> "AdaptiveSResdRVFL":
        """Fit full model from 1D raw time series."""
        series = np.asarray(series, dtype=float).ravel()
        if len(series) <= self.window_size + 5:
            raise ValueError("Series too short for the configured window_size.")

        series_scaled = self.scaler.fit_transform(series.reshape(-1, 1)).ravel()

        x_windows, y = make_sliding_windows(series_scaled, self.window_size)
        x_feat, _, _, _ = decompose_batch(x_windows, period=self.period, use_stl=self.use_stl)
        gate_features = build_gate_features(x_windows)

        self.blocks = []
        rng = np.random.default_rng(self.random_state)

        for bi in range(self.n_blocks):
            block = ResdRVFLBlock(
                n_layers=self.n_layers,
                hidden_units=self.hidden_units,
                reg_lambda=self.reg_lambda,
                forgetting_factor=self.forgetting_factor,
                gate_input_dim=gate_features.shape[1],
                gate_hidden_dim=8,
                random_state=int(rng.integers(0, 1_000_000)),
            )
            pred_block = block.fit(x_feat, y, gate_features)
            mse = float(np.mean((pred_block - y) ** 2) + EPS)

            self.blocks.append(block)
            self.block_mse_[bi] = mse

        self.is_fitted_ = True
        return self

    def _prepare_inference_inputs(
        self,
        x_raw: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before prediction.")

        x_raw = np.asarray(x_raw, dtype=float)
        if x_raw.ndim == 1:
            if len(x_raw) != self.window_size:
                raise ValueError(f"Expected {self.window_size} values, got {len(x_raw)}.")
            x_raw = x_raw.reshape(1, -1)

        if x_raw.shape[1] != self.window_size:
            raise ValueError(
                f"Input window length mismatch: expected {self.window_size}, got {x_raw.shape[1]}."
            )

        x_scaled = self.scaler.transform(x_raw.reshape(-1, 1)).reshape(x_raw.shape)
        x_feat, trend, seasonal, resid = decompose_batch(
            x_scaled,
            period=self.period,
            use_stl=self.use_stl,
        )
        gate_features = build_gate_features(x_scaled)
        return x_scaled, x_feat, gate_features, trend, seasonal, resid

    def predict_with_uncertainty(self, x_raw: np.ndarray) -> Dict[str, np.ndarray]:
        """Predict point forecast + uncertainty interval + attention weights."""
        _, x_feat, gate_features, trend, seasonal, resid = self._prepare_inference_inputs(x_raw)

        block_preds = []
        block_alphas = []
        for block in self.blocks:
            pred_b, alpha_b = block.predict(x_feat, gate_features)
            block_preds.append(pred_b)
            block_alphas.append(alpha_b)

        block_preds = np.column_stack(block_preds)  # (n_samples, n_blocks)
        block_alphas = np.column_stack(block_alphas)

        attn = self._attention_weights()
        point_pred_scaled = block_preds @ attn

        block_std = np.std(block_preds, axis=1)
        lower_scaled = point_pred_scaled - 1.96 * block_std
        upper_scaled = point_pred_scaled + 1.96 * block_std

        point_pred = self.scaler.inverse_transform(point_pred_scaled.reshape(-1, 1)).ravel()
        lower = self.scaler.inverse_transform(lower_scaled.reshape(-1, 1)).ravel()
        upper = self.scaler.inverse_transform(upper_scaled.reshape(-1, 1)).ravel()

        # Keep inter-block predictions in original scale for reporting.
        block_preds_orig = self.scaler.inverse_transform(block_preds.reshape(-1, 1)).reshape(block_preds.shape)

        decomp_summaries = [
            summarize_decomposition_components(trend[i], seasonal[i], resid[i])
            for i in range(trend.shape[0])
        ]

        return {
            "point_forecast": point_pred,
            "ci_lower": lower,
            "ci_upper": upper,
            "attention_weights": attn,
            "block_predictions": block_preds_orig,
            "block_std_scaled": block_std,
            "gate_alpha_per_block": block_alphas,
            "gate_alpha_mean": np.mean(block_alphas, axis=1),
            "decomposition_summary": np.array(decomp_summaries, dtype=object),
        }

    def update(self, x_raw_window: np.ndarray, y_actual_raw: float) -> Dict[str, np.ndarray]:
        """Online one-step update using RLS without full retraining."""
        x_raw_window = np.asarray(x_raw_window, dtype=float).ravel()
        if len(x_raw_window) != self.window_size:
            raise ValueError(
                f"update expects a window of length {self.window_size}, got {len(x_raw_window)}."
            )

        # Transform window + actual value with the fixed training scaler.
        x_scaled, x_feat, gate_features, _, _, _ = self._prepare_inference_inputs(x_raw_window)
        y_scaled = float(self.scaler.transform(np.array([[float(y_actual_raw)]])).ravel()[0])

        block_preds_before = np.zeros(self.n_blocks, dtype=float)

        for bi, block in enumerate(self.blocks):
            pred_before, _ = block.predict(x_feat, gate_features)
            block_preds_before[bi] = float(pred_before[0])

            _ = block.update_one(
                x_feat=x_feat[0],
                gate_feat=gate_features[0],
                y_true=y_scaled,
            )

            err2 = (y_scaled - block_preds_before[bi]) ** 2
            self.block_mse_[bi] = (
                self.performance_decay * self.block_mse_[bi]
                + (1.0 - self.performance_decay) * float(err2)
            )

        attn = self._attention_weights()
        pred_before_scaled = float(block_preds_before @ attn)
        pred_before = float(self.scaler.inverse_transform([[pred_before_scaled]])[0, 0])

        return {
            "prediction_before_update": np.array([pred_before]),
            "attention_weights": attn,
            "block_mse": self.block_mse_.copy(),
        }


def _find_candidate_aemo_csv(search_root: Path) -> Optional[Path]:
    """Find a likely AEMO CSV file in or below search_root."""
    candidates = list(search_root.rglob("*.csv"))
    for path in candidates:
        name = path.name.lower()
        if "aemo" in name or "electric" in name or "demand" in name:
            return path
    return None


def load_aemo_sample_series(
    csv_path: Optional[str] = None,
    region: str = "QLD",
    max_rows: int = 4000,
) -> Tuple[np.ndarray, Dict[str, str]]:
    """Load AEMO-like series from CSV; fallback to synthetic if unavailable."""
    resolved_path: Optional[Path] = None

    if csv_path:
        p = Path(csv_path)
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(
                f"CSV path not found: {csv_path}. "
                "Provide a valid AEMO CSV path or omit --csv-path to use synthetic fallback."
            )
        resolved_path = p
    else:
        resolved_path = _find_candidate_aemo_csv(Path.cwd())

    if resolved_path is not None:
        df = pd.read_csv(resolved_path)

        # Standardize column names for robust matching.
        rename_map = {c: c.strip().upper() for c in df.columns}
        df = df.rename(columns=rename_map)

        datetime_cols = [c for c in df.columns if "DATE" in c or "TIME" in c]
        if datetime_cols:
            dt_col = datetime_cols[0]
            df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
            df = df.sort_values(dt_col)

        if "REGIONID" in df.columns:
            df = df[df["REGIONID"].astype(str).str.upper() == region.upper()]

        preferred_value_cols = [
            "TOTALDEMAND",
            "DEMAND",
            "VALUE",
            "MW",
            "RRP",
        ]
        value_col = None
        for c in preferred_value_cols:
            if c in df.columns:
                value_col = c
                break

        if value_col is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                raise ValueError(f"No numeric columns found in {resolved_path}.")
            value_col = numeric_cols[0]

        series = df[value_col].astype(float).dropna().to_numpy()
        if max_rows > 0:
            series = series[:max_rows]

        if len(series) < 100:
            raise ValueError(
                f"Loaded too few points ({len(series)}). Check CSV format/region filter."
            )

        meta = {
            "source": str(resolved_path),
            "value_column": value_col,
            "region": region,
            "rows": str(len(series)),
        }
        return series, meta

    # Synthetic fallback preserving AEMO-like daily/weekly seasonality structure.
    n = max(1200, max_rows)
    t = np.arange(n, dtype=float)
    rng = np.random.default_rng(123)

    baseline = 5200.0
    daily = 280.0 * np.sin(2.0 * np.pi * t / 48.0)
    weekly = 140.0 * np.sin(2.0 * np.pi * t / (48.0 * 7.0))
    slow_trend = 0.05 * t
    noise = rng.normal(0.0, 35.0, size=n)

    series = baseline + daily + weekly + slow_trend + noise
    meta = {
        "source": "synthetic_fallback",
        "value_column": "TOTALDEMAND",
        "region": region,
        "rows": str(len(series)),
    }
    return series.astype(float), meta


def _parse_history_input(raw_text: str, expected_len: int) -> np.ndarray:
    parts = [p.strip() for p in raw_text.split(",") if p.strip()]
    values = np.array([float(p) for p in parts], dtype=float)
    if len(values) != expected_len:
        raise ValueError(f"Expected {expected_len} values, received {len(values)}.")
    return values


def run_cli(args: argparse.Namespace) -> None:
    """Train model and start interactive forecast/update loop."""
    series, meta = load_aemo_sample_series(
        csv_path=args.csv_path,
        region=args.region,
        max_rows=args.max_rows,
    )

    print("\n=== Data Loaded ===")
    print(f"Source           : {meta['source']}")
    print(f"Region           : {meta['region']}")
    print(f"Value column     : {meta['value_column']}")
    print(f"Rows             : {meta['rows']}")

    model = AdaptiveSResdRVFL(
        window_size=args.window_size,
        n_blocks=args.n_blocks,
        n_layers=args.n_layers,
        hidden_units=[args.hidden_units] * args.n_layers,
        reg_lambda=args.reg_lambda,
        period=args.period,
        use_stl=(not args.disable_stl),
        attention_temperature=args.attn_temp,
        attention_error_power=args.attn_power,
        forgetting_factor=args.forgetting,
        performance_decay=args.perf_decay,
        random_state=args.random_state,
    )

    model.fit(series)
    print("\n=== Model Ready ===")
    print(
        f"AdaptiveSResdRVFL fitted with {args.n_blocks} blocks, "
        f"{args.n_layers} layers/block, window={args.window_size}."
    )

    print("\nInteractive mode:")
    print(f"- Enter exactly {args.window_size} comma-separated values to forecast next step.")
    print("- Enter 'exit' to quit.")

    while True:
        user_text = input("\nEnter historical window: ").strip()
        if user_text.lower() in {"exit", "quit", "q"}:
            print("Session ended.")
            break
        if not user_text:
            print(
                f"Input error: no values provided. Enter exactly "
                f"{args.window_size} comma-separated values."
            )
            continue

        try:
            history = _parse_history_input(user_text, expected_len=args.window_size)
        except Exception as exc:
            print(f"Input error: {exc}")
            continue

        try:
            out = model.predict_with_uncertainty(history)
        except Exception as exc:
            print(f"Prediction error: {exc}")
            continue

        point = float(out["point_forecast"][0])
        lo = float(out["ci_lower"][0])
        hi = float(out["ci_upper"][0])
        attn = out["attention_weights"]
        summary = out["decomposition_summary"][0]
        s_mean = float(out["gate_alpha_mean"][0])
        s_blocks = out["gate_alpha_per_block"][0]

        print("\n=== Forecast Report ===")
        print(f"Input Summary      : Last {args.window_size} steps received")
        print(
            "Decomposition      : "
            f"Trend: {summary['trend']} / "
            f"Seasonal: {summary['seasonal']} / "
            f"Residual: {summary['residual']}"
        )
        print(f"Point Forecast     : {point:.4f}")
        print(f"95% Confidence     : [{lo:.4f}, {hi:.4f}]")
        attn_text = ", ".join([f"Block {i + 1}: {w:.4f}" for i, w in enumerate(attn)])
        print(f"Attention Weights  : {attn_text}")
        s_text = ", ".join([f"Block {i + 1}: {v:.4f}" for i, v in enumerate(s_blocks)])
        print(f"Learned Scaling s  : Mean {s_mean:.4f} ({s_text})")

        update_txt = input("Enter actual next value to update model (blank to skip): ").strip()
        if update_txt:
            try:
                y_true = float(update_txt)
                update_out = model.update(history, y_true)
                print("Update status      : Online RLS update applied.")
                print(
                    "Updated Attention  : "
                    + ", ".join(
                        [
                            f"Block {i + 1}: {w:.4f}"
                            for i, w in enumerate(update_out["attention_weights"])
                        ]
                    )
                )
            except Exception as exc:
                print(f"Update error: {exc}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Adaptive Attention-based SResdRVFL for Time-Series Forecasting"
    )
    parser.add_argument("--csv-path", type=str, default=None, help="Path to AEMO CSV file")
    parser.add_argument("--region", type=str, default="QLD", help="Region filter (if REGIONID exists)")
    parser.add_argument("--max-rows", type=int, default=4000, help="Maximum rows to load")

    parser.add_argument("--window-size", type=int, default=50, help="Sliding window length")
    parser.add_argument("--n-blocks", type=int, default=5, help="Number of ResdRVFL blocks")
    parser.add_argument("--n-layers", type=int, default=3, help="Number of deep layers per block")
    parser.add_argument("--hidden-units", type=int, default=64, help="Hidden units per layer")

    parser.add_argument("--reg-lambda", type=float, default=1e-3, help="Ridge regularization")
    parser.add_argument("--period", type=int, default=24, help="Seasonal period for decomposition")
    parser.add_argument("--disable-stl", action="store_true", help="Disable STL even if statsmodels exists")

    parser.add_argument(
        "--attn-temp",
        type=float,
        default=0.08,
        help="Attention softmax temperature (lower => sharper block separation)",
    )
    parser.add_argument(
        "--attn-power",
        type=float,
        default=2.0,
        help="Nonlinear penalty power on relative block MSE (higher => more sensitive)",
    )
    parser.add_argument("--forgetting", type=float, default=0.995, help="RLS forgetting factor")
    parser.add_argument(
        "--perf-decay",
        type=float,
        default=0.85,
        help="EMA decay for block performance (lower => faster adaptation)",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")

    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    run_cli(args)


"""
Adaptive Attention-based Stacked Residual RVFL Network with Uncertainty Estimation.

This module implements an adaptive variant of SResdRVFL for univariate time-series
forecasting. Core features:
- Max-Min normalization
- Sliding-window supervised framing
- Multi-resolution decomposition (STL when available, fallback decomposition otherwise)
- Stacked Residual Deep RVFL blocks with fixed random hidden layers
- Closed-form output weights via Moore-Penrose pseudo-inverse
- Adaptive residual scaling with a tiny MLP gate (meta-learner)
- Attention-based aggregation across ensemble blocks
- Predictive uncertainty via inter-block variance (95% confidence interval)
- Online updates using Recursive Least Squares (RLS)
- dependency: statsmodels (for STL decomposition)
"""
