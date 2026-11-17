"""Next-day PM2.5 forecasting for Astana, Kazakhstan.

The package implements a reproducible, leakage-free machine learning backend:
data acquisition from the OpenWeatherMap Air Pollution API, validation,
cleaning, feature engineering at a fixed 18:00 local issue time, four
regression models plus a persistence baseline, evaluation, and a
command-line next-day forecast.
"""

__version__ = "1.0.0"
