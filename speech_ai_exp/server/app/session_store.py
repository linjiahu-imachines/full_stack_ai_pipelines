from __future__ import annotations

import json
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TurnRecord:
    turn_index: int
    created_at: str
    user_transcript: str
    assistant_reply: str
    user_audio: str
    reply_audio: str
    timings: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TurnRecord:
        return cls(
            turn_index=int(data["turn_index"]),
            created_at=str(data["created_at"]),
            user_transcript=str(data["user_transcript"]),
            assistant_reply=str(data["assistant_reply"]),
            user_audio=str(data["user_audio"]),
            reply_audio=str(data["reply_audio"]),
            timings=dict(data.get("timings") or {}),
        )


@dataclass
class Session:
    session_id: str
    created_at: str
    updated_at: str
    turns: list[TurnRecord] = field(default_factory=list)

    def history_messages(self) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        for t in self.turns:
            if t.user_transcript.strip():
                out.append({"role": "user", "content": t.user_transcript.strip()})
            if t.assistant_reply.strip():
                out.append({"role": "assistant", "content": t.assistant_reply.strip()})
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "turns": [t.to_dict() for t in self.turns],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Session:
        return cls(
            session_id=str(data["session_id"]),
            created_at=str(data["created_at"]),
            updated_at=str(data["updated_at"]),
            turns=[TurnRecord.from_dict(t) for t in data.get("turns") or []],
        )


class SessionStore:
    def __init__(self, root: Path) -> None:
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, Session] = {}
        self._lock = threading.Lock()

    def _session_dir(self, session_id: str) -> Path:
        return self._root / session_id

    def _session_json(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "session.json"

    def _save(self, session: Session) -> None:
        d = self._session_dir(session.session_id)
        d.mkdir(parents=True, exist_ok=True)
        self._session_json(session.session_id).write_text(
            json.dumps(session.to_dict(), indent=2),
            encoding="utf-8",
        )

    def _load_from_disk(self, session_id: str) -> Session | None:
        path = self._session_json(session_id)
        if not path.is_file():
            return None
        return Session.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def create(self) -> Session:
        sid = uuid.uuid4().hex
        now = _utc_now()
        session = Session(session_id=sid, created_at=now, updated_at=now)
        with self._lock:
            self._cache[sid] = session
            self._save(session)
        return session

    def get(self, session_id: str) -> Session | None:
        with self._lock:
            if session_id in self._cache:
                return self._cache[session_id]
        session = self._load_from_disk(session_id)
        if session is not None:
            with self._lock:
                self._cache[session_id] = session
        return session

    def update(self, session: Session) -> None:
        session.updated_at = _utc_now()
        with self._lock:
            self._cache[session.session_id] = session
            self._save(session)

    def delete(self, session_id: str) -> bool:
        with self._lock:
            self._cache.pop(session_id, None)
        d = self._session_dir(session_id)
        if not d.exists():
            return False
        for p in sorted(d.glob("*"), reverse=True):
            if p.is_file():
                p.unlink()
        d.rmdir()
        return True

    def session_audio_dir(self, session_id: str) -> Path:
        d = self._session_dir(session_id)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def list_sessions(self) -> list[dict[str, Any]]:
        summaries: list[dict[str, Any]] = []
        if not self._root.is_dir():
            return summaries
        for path in self._root.iterdir():
            if not path.is_dir():
                continue
            sid = path.name
            session = self.get(sid)
            if session is None:
                continue
            first_user = ""
            last_asst = ""
            if session.turns:
                first_user = (session.turns[0].user_transcript or "")[:120]
                last_asst = (session.turns[-1].assistant_reply or "")[:120]
            summaries.append(
                {
                    "session_id": session.session_id,
                    "created_at": session.created_at,
                    "updated_at": session.updated_at,
                    "turn_count": len(session.turns),
                    "preview_first_user": first_user,
                    "preview_last_assistant": last_asst,
                }
            )
        summaries.sort(key=lambda s: s["updated_at"], reverse=True)
        return summaries
