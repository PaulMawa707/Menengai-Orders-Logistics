import os
import secrets
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=False)

__all__ = ["BASE_DIR", "config", "Config"]


def _require(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            f"Copy .env.example to .env and set all values."
        )
    return value


class Config:
    def __init__(self):
        self.SECRET_KEY = _require("SECRET_KEY")
        self.APP_USERNAME = _require("APP_USERNAME")
        self.APP_PASSWORD = _require("APP_PASSWORD")
        self.WIALON_TOKEN = _require("WIALON_TOKEN")
        self.WIALON_RESOURCE_ID = int(_require("WIALON_RESOURCE_ID"))
        self.MAX_CONTENT_LENGTH = 32 * 1024 * 1024
        self.CONTROLTECH_URL = os.getenv(
            "CONTROLTECH_URL", "https://www.controltech-ea.com/"
        ).strip()


config = Config()


def get_auth_credentials():
    load_dotenv(BASE_DIR / ".env", override=True)
    return (
        os.getenv("APP_USERNAME", "").strip(),
        os.getenv("APP_PASSWORD", "").strip(),
    )


def verify_login(username: str, password: str) -> bool:
    expected_user, expected_pass = get_auth_credentials()
    username = username.strip()
    password = password.strip()
    if not expected_user or not expected_pass:
        return False
    return secrets.compare_digest(username, expected_user) and secrets.compare_digest(
        password, expected_pass
    )
