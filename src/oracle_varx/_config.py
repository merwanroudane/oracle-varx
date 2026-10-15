"""Experiment configuration for ORACLE-VARX.

Mirrors Table 10, Panel A of Tan et al. (2026): the window sizes, the significance
grid and the refit interval that define a run. Every default is the paper's value.

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
GitHub: https://github.com/merwanroudane
"""

from __future__ import annotations

from dataclasses import dataclass, replace

__all__ = ["OracleVARXConfig", "ALPHA_GRID"]

#: The significance grid of Table 10, Panel A.
ALPHA_GRID: tuple[float, ...] = (0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)


@dataclass(frozen=True)
class OracleVARXConfig:
    """Window sizes and tuning grid for one ORACLE-VARX run.

    Parameters
    ----------
    p_max : int, default 5
        Maximum lag order considered. Must exceed the true maximum causal lag for
        assumption (M) of the paper to hold; the paper uses 5 on the synthetic
        benchmark and 10 on the ETF application.
    alpha_grid : tuple of float, default :data:`ALPHA_GRID`
        The Stage-1 significance levels ACLE searches over (eq. 10).
    ols_window : int, default 200
        The rolling window ``T_w`` the second-stage OLS (eq. 7) is run on.
    train_window : int or None, default None
        Rows used to fit the first-stage learners in each block. ``None`` means
        ``ols_window``, which is what the paper does.
    block : int, default 20
        Block length ``n_test`` of the blocked cross-fitting split (§3.2), and the
        refit interval: fold ``k`` trains on ``[s_k, s_k + train_window)``, tests on
        the next ``block`` rows, and ``s_{k+1} = s_k + block``.
    validation_days : int, default 20
        Length ``v`` of the trailing validation window of eq. (9).
    buffer : int, default 0
        Observations dropped between each training fold and its test fold. The
        paper deliberately uses adjacent folds (``0``, Remark 1); a positive value
        gives the buffer zones of Ballinari and Wehrli (2024).
    q : float, default 0.05
        Target false discovery rate of the BH edge-discovery step (§3.4, D.4).
    lag_offset : int, default 5
        Extra rows of history kept in front of each window so the deepest lag can
        be built. The paper's implementation uses 5 on the synthetic benchmark and
        10 on the ETF application.

    Notes
    -----
    ``lookback`` is the history each output day needs and is derived, not set.

    Examples
    --------
    >>> from oracle_varx import OracleVARXConfig
    >>> cfg = OracleVARXConfig.synthetic()
    >>> cfg.p_max, cfg.ols_window, cfg.block
    (5, 200, 20)
    >>> OracleVARXConfig.financial().lookback
    1018
    """

    p_max: int = 5
    alpha_grid: tuple[float, ...] = ALPHA_GRID
    ols_window: int = 200
    train_window: int | None = None
    block: int = 20
    validation_days: int = 20
    buffer: int = 0
    q: float = 0.05
    lag_offset: int = 5

    def __post_init__(self) -> None:
        if self.p_max < 1:
            raise ValueError(f"p_max must be >= 1, got {self.p_max}")
        if self.ols_window < 1:
            raise ValueError(f"ols_window must be >= 1, got {self.ols_window}")
        if self.block < 1:
            raise ValueError(f"block must be >= 1, got {self.block}")
        if self.validation_days < 1:
            raise ValueError(
                f"validation_days must be >= 1, got {self.validation_days}"
            )
        if not 0.0 < self.q < 1.0:
            raise ValueError(f"q must lie in (0, 1), got {self.q}")
        if not self.alpha_grid:
            raise ValueError("alpha_grid must hold at least one level")
        if any(not 0.0 < a < 1.0 for a in self.alpha_grid):
            raise ValueError(f"every alpha must lie in (0, 1), got {self.alpha_grid}")
        if self.buffer < 0:
            raise ValueError(f"buffer must be >= 0, got {self.buffer}")

    # ------------------------------------------------------------------ presets
    @classmethod
    def synthetic(cls) -> OracleVARXConfig:
        """The paper's synthetic-benchmark column of Table 10.

        Examples
        --------
        >>> OracleVARXConfig.synthetic().validation_days
        20
        """
        return cls(
            p_max=5,
            ols_window=200,
            block=20,
            validation_days=20,
            lag_offset=5,
        )

    @classmethod
    def financial(cls) -> OracleVARXConfig:
        """The paper's ETF-application column of Table 10.

        Examples
        --------
        >>> OracleVARXConfig.financial().p_max
        10
        """
        return cls(
            p_max=10,
            ols_window=504,
            block=21,
            validation_days=21,
            lag_offset=10,
        )

    def replace(self, **changes) -> OracleVARXConfig:
        """Return a copy with some fields changed.

        Examples
        --------
        >>> OracleVARXConfig.synthetic().replace(q=0.10).q
        0.1
        """
        return replace(self, **changes)

    # --------------------------------------------------------------- properties
    @property
    def n_train(self) -> int:
        """Rows used to fit the first stage in each block."""
        return self.ols_window if self.train_window is None else self.train_window

    @property
    def lookback(self) -> int:
        """History each output day needs: first stage + second stage + lag offset."""
        return self.n_train + self.ols_window + self.lag_offset

    @property
    def lookback_ols(self) -> int:
        """History a plain VAR/VARX output day needs (no first stage)."""
        return self.ols_window + self.lag_offset

    def summary(self) -> str:
        """The configuration as a printable block, in the style of Table 10.

        Examples
        --------
        >>> print(OracleVARXConfig.synthetic().summary())  # doctest: +ELLIPSIS
        ORACLE-VARX configuration...
        """
        grid = ", ".join(f"{a:g}" for a in self.alpha_grid)
        lines = [
            "ORACLE-VARX configuration",
            "=" * 58,
            f"  Maximum lag order (p_max)        : {self.p_max:>10d}",
            f"  Second-stage window (T_w)        : {self.ols_window:>10d}",
            f"  First-stage training window      : {self.n_train:>10d}",
            f"  Block length / refit interval    : {self.block:>10d}",
            f"  Validation days (v)              : {self.validation_days:>10d}",
            f"  Buffer between folds (k_T)       : {self.buffer:>10d}",
            f"  Edge FDR level (q)               : {self.q:>10.3f}",
            f"  Lag offset                       : {self.lag_offset:>10d}",
            f"  Required lookback per output day : {self.lookback:>10d}",
            "-" * 58,
            f"  alpha-grid: {{{grid}}}",
            "=" * 58,
        ]
        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.summary()
