# =============================
# backend/app/mcp/session.py
# =============================
from __future__ import annotations
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List


class Session:
    def __init__(self, session_id: str):
        self.id = session_id
        self.created_at = datetime.utcnow()
        self.messages: List[Dict[str, Any]] = []
        self.meta: Dict[str, Any] = {}


class SessionStore:
    def __init__(self, ttl_minutes: int = 120):
        self._ttl = timedelta(minutes=ttl_minutes)
        self._data: dict[str, Session] = {}

    def open(self, session_id: str | None) -> tuple[Session, bool]:
        sid = session_id or str(uuid.uuid4())
        created = False
        if sid not in self._data:
            self._data[sid] = Session(sid)
            created = True
        return self._data[sid], created

    def get(self, session_id: str) -> Session | None:
        return self._data.get(session_id)

    def append(self, session_id: str, role: str, content: str, meta: Dict[str, Any]):
        s = self._data[session_id]
        s.messages.append({"role": role, "content": content, "meta": meta, "ts": datetime.utcnow().isoformat()})

    def close(self, session_id: str):
        self._data.pop(session_id, None)


store = SessionStore()