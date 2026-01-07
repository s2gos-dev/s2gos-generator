"""Snow seasonality model using synthetic temperature and logistic probability.

Temperature: T(φ,z,d) = T_R - β·φ + A(φ)cos(2π/365·(d-d_max)) - Γ·z
Probability: P_snow = 1/(1 + exp((T-T_c)/σ)) for T ≤ 0.1°C

Supports January and July calculations for Northern/Southern hemispheres.
"""

import logging
from typing import Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

from ..core.config import Month

# Temperature Model Constants
T_R = 26.0  # Reference temperature at equator, sea level (°C)
BETA = 0.62  # Latitudinal gradient (°C/degree)
LAPSE_RATE = 0.0095  # Temperature lapse rate: 9.5°C/km -> 0.0095 °C/m

# Seasonal Amplitude Interpolation
A_HIGH_LAT = 18.0  # Amplitude at high latitude (°C)
LAT_HIGH = 60.0  # Reference high latitude (degrees)
A_LOW_LAT = 3.0  # Amplitude at low latitude (°C)
LAT_LOW = 10.0  # Reference low latitude (degrees)

# Phase Constants
D_MAX_NH = 202  # Day of maximum temperature in Northern Hemisphere (~July 21)
D_MAX_SH = 20  # Day of maximum temperature in Southern Hemisphere (~Jan 20)

# Snow Probability Model
T_C = 1.0  # 50% rain-snow transition temperature (°C)
SIGMA = 1.5  # Logistic function width (°C)
HARD_FREEZE_LIMIT = 0.1  # Temperature above which snow is impossible (°C)



def get_day_of_year(month: Month) -> int:
    """Map Month enum to approximate day of year."""
    if month == Month.JANUARY:
        return 20
    elif month == Month.JULY:
        return 202
    else:
        return 1

def apply_spatial_smoothing(
    data: np.ndarray,
    sigma: float = 10.0
) -> np.ndarray:
    """Apply Gaussian spatial smoothing."""
    if sigma <= 0:
        return data
    return gaussian_filter(data, sigma=sigma, mode='nearest')


# --- Core Logic ---

def calculate_seasonal_amplitude(abs_lat: np.ndarray) -> np.ndarray:
    """Calculate seasonal amplitude A(φ) via linear interpolation."""
    slope = (A_HIGH_LAT - A_LOW_LAT) / (LAT_HIGH - LAT_LOW)
    amplitude = A_LOW_LAT + slope * (abs_lat - LAT_LOW)
    return np.clip(amplitude, A_LOW_LAT, A_HIGH_LAT)


def calculate_temperature_field(
    latitudes: np.ndarray,
    elevations: np.ndarray,
    day_of_year: int,
) -> np.ndarray:
    """Calculate temperature: T(φ,z,d) = T_R - β·φ + A(φ)cos(2π/365·(d-d_max)) - Γ·z"""
    abs_lat = np.abs(latitudes)
    lat_term = -BETA * abs_lat
    amplitude = calculate_seasonal_amplitude(abs_lat)
    d_max_map = np.where(latitudes >= 0, D_MAX_NH, D_MAX_SH)
    phase = (2.0 * np.pi / 365.0) * (day_of_year - d_max_map)
    seasonal_term = amplitude * np.cos(phase)
    elevation_term = -LAPSE_RATE * elevations
    return T_R + lat_term + seasonal_term + elevation_term


def calculate_snow_probability_map(
    latitudes: np.ndarray,
    elevations: np.ndarray,
    day_of_year: int,
    smooth_sigma: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate snow probability using logistic model: P = 1/(1 + exp((T-T_c)/σ)) for T ≤ 0.1°C

    Returns: (probabilities, temperatures)
    """
    temperatures = calculate_temperature_field(latitudes, elevations, day_of_year)
    probabilities = np.zeros_like(temperatures, dtype=np.float32)

    cold_mask = temperatures <= HARD_FREEZE_LIMIT
    if np.any(cold_mask):
        cold_temps = temperatures[cold_mask]
        exponent = (cold_temps - T_C) / SIGMA
        probabilities[cold_mask] = 1.0 / (1.0 + np.exp(exponent))

    if smooth_sigma > 0:
        probabilities = apply_spatial_smoothing(probabilities, sigma=smooth_sigma)

    return probabilities, temperatures

