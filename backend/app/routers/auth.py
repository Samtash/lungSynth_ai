"""
Google Sign-In verification.

If LUNGSYNTH_GOOGLE_CLIENT_ID is set, the ID token from Google Identity
Services is cryptographically verified via google-auth. If it is unset,
the backend runs in DEV mode: it decodes the JWT payload without
verifying its signature so you can develop the UI without setting up
OAuth credentials. DEV mode is never safe to expose publicly.
"""

from __future__ import annotations

import base64
import json
import logging

from fastapi import APIRouter, HTTPException

from app.config import settings
from app.schemas import GoogleAuthRequest, StoredUser

router = APIRouter(prefix="/api/auth", tags=["auth"])
logger = logging.getLogger("lungsynth.auth")


def _initials(name: str) -> str:
    parts = [p for p in name.split() if p]
    return "".join(p[0].upper() for p in parts[-2:]) or "U"


def _decode_unverified(credential: str) -> dict:
    try:
        payload_b64 = credential.split(".")[1]
        padded = payload_b64 + "=" * (-len(payload_b64) % 4)
        return json.loads(base64.urlsafe_b64decode(padded))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Malformed credential: {exc}") from exc


@router.post("/google", response_model=StoredUser)
def google_sign_in(body: GoogleAuthRequest) -> StoredUser:
    if settings.google_client_id:
        from google.auth.transport import requests as google_requests
        from google.oauth2 import id_token as google_id_token

        try:
            payload = google_id_token.verify_oauth2_token(
                body.credential, google_requests.Request(), settings.google_client_id
            )
        except ValueError as exc:
            raise HTTPException(status_code=401, detail=f"Invalid Google credential: {exc}") from exc
    else:
        logger.warning(
            "LUNGSYNTH_GOOGLE_CLIENT_ID is not set — accepting Google credential "
            "WITHOUT signature verification (dev mode only)."
        )
        payload = _decode_unverified(body.credential)

    email = payload.get("email", "unknown@example.com")
    name = payload.get("name", email.split("@")[0])
    return StoredUser(name=name, email=email, initials=_initials(name))
