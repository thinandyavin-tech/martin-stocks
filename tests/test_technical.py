import numpy as np
import pandas as pd

from utils.technical import (
    calculate_obv, calculate_rsi, range_position, relative_strength_vs_spy, volume_ratio,
)


def test_rsi_defaults_to_neutral_without_enough_data():
    assert calculate_rsi(pd.Series([1.0, 2.0, 3.0]), period=14) == 50.0


def test_rsi_extremes_follow_trend():
    up = pd.Series(np.linspace(100, 130, 40) + np.r_[0, np.tile([0.5, -0.2], 20)][:40])
    down = pd.Series(np.linspace(130, 100, 40) + np.r_[0, np.tile([-0.5, 0.2], 20)][:40])
    assert calculate_rsi(up) > 70
    assert calculate_rsi(down) < 30


def test_range_position():
    assert range_position(150, 100, 200) == 50.0
    assert range_position(100, 100, 200) == 0.0
    assert range_position(200, 100, 200) == 100.0
    assert range_position(5, 5, 5) == 50.0   # flat range doesn't divide by zero


def test_volume_ratio():
    vol = pd.Series([100.0] * 20 + [300.0])
    assert volume_ratio(vol) == 3.0
    assert volume_ratio(pd.Series([1.0, 2.0])) == 1.0


def test_relative_strength_vs_spy():
    stock = pd.Series(np.linspace(100, 120, 90))   # +20%
    spy = pd.Series(np.linspace(100, 110, 90))     # +10%
    assert relative_strength_vs_spy(stock, spy, period=90) == 2.0


def test_obv_accumulates_volume_by_direction():
    close = pd.Series([10.0, 11.0, 10.5, 12.0])
    volume = pd.Series([100.0, 200.0, 50.0, 300.0])
    assert calculate_obv(close, volume).tolist() == [0.0, 200.0, 150.0, 450.0]
