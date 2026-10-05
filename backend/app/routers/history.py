from __future__ import annotations

from fastapi import APIRouter, Query

from app import storage
from app.schemas import HistoryEntry

router = APIRouter(prefix="/api/history", tags=["history"])


@router.get("", response_model=list[HistoryEntry])
def list_history(user_email: str | None = Query(default=None)) -> list[HistoryEntry]:
    rows = storage.list_sessions(user_email=user_email)
    return [
        HistoryEntry(
            id=r["id"],
            sessionId=r["session_code"],
            createdAt=r["created_at"],
            phases=r["phases"],
            processingSeconds=r["processing_seconds"],
            status=r["status"],
        )
        for r in rows
    ]


@router.delete("")
def clear_history(user_email: str | None = Query(default=None)):
    storage.clear_sessions(user_email=user_email)
    return {"ok": True}
