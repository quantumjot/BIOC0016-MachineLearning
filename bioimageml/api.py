import hashlib
import os
import uuid

import requests

from . import base

API_ANONYMOUS_IDENTIFIER = uuid.uuid4()
API_HASH = "5f19013a94fbc30cd32b5425843746b1"
PART_TOKEN = "BlHuBbgaghHcWh1tkDrIw0XC8bMZxr88R86REAyw"


def validate_api_token(token: str) -> None:
    """Validate the API token."""
    full_token = token + PART_TOKEN
    md5 = hashlib.md5()
    md5.update(full_token.encode("utf-8"))
    token_hash = str(md5.hexdigest())
    if token_hash != API_HASH:
        raise ValueError("API token is invalid.")
    os.environ["API_TOKEN"] = full_token


def submit_annotation(annotation: base.ImageWithAnnotation) -> None:
    """Submit a response to the form."""

    if not isinstance(annotation, base.ImageWithAnnotation):
        raise TypeError

    API_TOKEN = os.environ.get("API_TOKEN", None)
    if API_TOKEN is None:
        raise ValueError("No API token has been provided.")

    data_json = {
        "entry.1863715764": annotation.identifier,
        "entry.389727206": annotation.hash,
        "entry.132579272": str(annotation.label.name).capitalize(),
        "entry.944720336": str(API_ANONYMOUS_IDENTIFIER),
    }

    url = f"https://docs.google.com/forms/d/e/{API_TOKEN}/formResponse"

    response = requests.post(url, data_json)

    if response.status_code != 200:
        raise ValueError("Something is wrong with the server.")
