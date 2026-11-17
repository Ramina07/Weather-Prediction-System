"""Configuration loading, path resolution, secret handling and logging.

The API key is read only from the ``OPENWEATHER_API_KEY`` environment variable
(optionally loaded from a ``.env`` file). It is never written to logs: a
logging filter redacts both the key itself and any ``appid=...`` query value.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

API_KEY_ENV = "OPENWEATHER_API_KEY"
PLACEHOLDER_KEYS = {"", "your_openweathermap_api_key_here", "...", "changeme", "none"}

REQUIRED_SECTIONS = ("project", "location", "api", "download", "paths",
                     "forecast", "pollutants", "splits", "training", "evaluation")


class ConfigError(ValueError):
    """Raised when config.yaml is missing or invalid."""


class MissingAPIKeyError(RuntimeError):
    """Raised when no usable OpenWeatherMap API key is configured."""


def find_project_root(start: Path | None = None) -> Path:
    """Return the closest parent directory that contains ``config.yaml``."""
    env_root = os.environ.get("AIR_QUALITY_ROOT")
    if env_root:
        return Path(env_root).resolve()
    here = (start or Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "config.yaml").exists():
            return candidate
    pkg_root = Path(__file__).resolve().parents[2]
    if (pkg_root / "config.yaml").exists():
        return pkg_root
    raise ConfigError("Could not locate config.yaml. Run commands from the ML_final folder "
                      "or set AIR_QUALITY_ROOT.")


@dataclass
class Config:
    """Thin wrapper around the parsed YAML with path helpers."""

    data: dict[str, Any]
    root: Path
    source: Path | None = None
    _validated: bool = field(default=False, repr=False)

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def path(self, key: str) -> Path:
        """Resolve ``paths.<key>`` relative to the project root."""
        try:
            rel = self.data["paths"][key]
        except KeyError as exc:
            raise ConfigError(f"paths.{key} is not defined in config.yaml") from exc
        p = Path(rel)
        return p if p.is_absolute() else self.root / p

    def ensure_dirs(self) -> None:
        for key, rel in self.data["paths"].items():
            p = self.path(key)
            (p if not p.suffix else p.parent).mkdir(parents=True, exist_ok=True)

    @property
    def tz(self) -> str:
        return self.data["location"]["timezone"]

    @property
    def seed(self) -> int:
        return int(self.data["project"]["random_seed"])


def validate_config(data: dict[str, Any]) -> None:
    """Basic structural validation with informative error messages."""
    if not isinstance(data, dict):
        raise ConfigError("config.yaml must contain a mapping at the top level.")
    missing = [s for s in REQUIRED_SECTIONS if s not in data]
    if missing:
        raise ConfigError(f"config.yaml is missing sections: {missing}")
    loc = data["location"]
    try:
        lat, lon = float(loc["latitude"]), float(loc["longitude"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigError("location.latitude/longitude must be numbers") from exc
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ConfigError("location coordinates are out of range")
    hour = data["forecast"].get("issue_hour_local")
    if not isinstance(hour, int) or not 1 <= hour <= 23:
        raise ConfigError("forecast.issue_hour_local must be an integer in [1, 23]")
    mvh = data["forecast"].get("target_min_valid_hours")
    if not isinstance(mvh, int) or not 1 <= mvh <= 24:
        raise ConfigError("forecast.target_min_valid_hours must be an integer in [1, 24]")
    for split in ("train", "validation", "test"):
        if split not in data["splits"] or len(data["splits"][split]) != 2:
            raise ConfigError(f"splits.{split} must be a [start, end] pair")
    tr, va, te = (data["splits"][s] for s in ("train", "validation", "test"))
    if not (tr[1] < va[0] and va[1] < te[0]):
        raise ConfigError("splits must be chronological and non-overlapping (train < validation < test)")
    if "pm2_5" not in data["pollutants"]:
        raise ConfigError("pollutants must include pm2_5")


def load_config(path: str | Path | None = None) -> Config:
    """Load and validate ``config.yaml``."""
    if path is None:
        root = find_project_root()
        path = root / "config.yaml"
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Configuration file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"config.yaml is not valid YAML: {exc}") from None
    validate_config(data)
    return Config(data=data, root=path.resolve().parent, source=path.resolve(), _validated=True)


# --------------------------------------------------------------------------- secrets
def get_api_key(root: Path | None = None, required: bool = True) -> str | None:
    """Return the API key from the environment or ``.env``; never log it."""
    try:
        from dotenv import load_dotenv
        env_file = (root or find_project_root()) / ".env"
        if env_file.exists():
            load_dotenv(env_file, override=False)
    except ImportError:  # pragma: no cover - python-dotenv is a hard requirement
        pass
    except ConfigError:
        pass
    key = os.environ.get(API_KEY_ENV, "").strip()
    if key.lower() in PLACEHOLDER_KEYS:
        if required:
            raise MissingAPIKeyError(
                f"{API_KEY_ENV} is not set. Copy .env.example to .env in the project root and "
                f"replace the placeholder with your OpenWeatherMap key "
                f"(line: {API_KEY_ENV}=<your key>).")
        return None
    register_secret(key)
    return key


_SECRETS: set[str] = set()
_APPID_RE = re.compile(r"(appid=)[^&\s'\"]+", re.IGNORECASE)


def register_secret(secret: str) -> None:
    if secret:
        _SECRETS.add(secret)


def redact(text: str) -> str:
    """Remove the API key and any appid query value from ``text``."""
    if not text:
        return text
    out = _APPID_RE.sub(r"\1***", str(text))
    for s in _SECRETS:
        out = out.replace(s, "***")
    return out


class RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401
        try:
            msg = record.getMessage()
        except Exception:  # pragma: no cover
            return True
        record.msg = redact(msg)
        record.args = ()
        return True


def setup_logging(cfg: Config | None = None, name: str = "pipeline", level: int = logging.INFO) -> logging.Logger:
    """Configure console + file logging with credential redaction."""
    logger = logging.getLogger("air_quality")
    logger.setLevel(level)
    if getattr(logger, "_configured", False):
        return logger
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    sh.addFilter(RedactingFilter())
    logger.addHandler(sh)
    if cfg is not None:
        log_dir = cfg.path("logs_dir")
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
        fh.setFormatter(fmt)
        fh.addFilter(RedactingFilter())
        logger.addHandler(fh)
    logger.propagate = False
    logger._configured = True  # type: ignore[attr-defined]
    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger("air_quality")
