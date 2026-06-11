from __future__ import annotations

import logging
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.audio_util import prepare_wav_for_asr
from app.config import ServerConfig
from app.pipeline_service import PipelineService
from app.session_store import Session, SessionStore, TurnRecord, _utc_now

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

cfg = ServerConfig.from_env()
sessions = SessionStore(cfg.sessions_dir)
pipeline = PipelineService(cfg.staged_config)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info("Starting pipeline load (this may take a minute on first run)...")
    pipeline.load()
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Intellengent Machines Agentic Assitant",
    description="Turn-based voice chatbot (Project 1: ASR → LLM → TTS)",
    lifespan=lifespan,
)


class SessionSummary(BaseModel):
    session_id: str
    created_at: str
    updated_at: str
    turn_count: int
    preview_first_user: str = ""
    preview_last_assistant: str = ""


class SessionCreated(BaseModel):
    session_id: str


class Timings(BaseModel):
    asr_s: float
    llm_ttft_s: float
    llm_generation_s: float
    tts_s: float


class TurnResponse(BaseModel):
    turn_index: int
    transcript: str
    reply_text: str
    reply_audio_url: str
    user_audio_url: str
    timings: Timings
    context_messages: int
    turn_count: int


class SessionResponse(BaseModel):
    session_id: str
    created_at: str
    updated_at: str
    turn_count: int
    turns: list[dict[str, Any]]


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok" if pipeline.ready else "loading",
        "pipeline_ready": pipeline.ready,
        "pipeline_error": pipeline.error,
        "config": str(cfg.staged_config),
    }


@app.get("/api/sessions", response_model=list[SessionSummary])
def list_sessions() -> list[SessionSummary]:
    return [SessionSummary(**s) for s in sessions.list_sessions()]


@app.post("/api/sessions", response_model=SessionCreated)
def create_session() -> SessionCreated:
    session = sessions.create()
    return SessionCreated(session_id=session.session_id)


@app.get("/api/sessions/{session_id}", response_model=SessionResponse)
def get_session(session_id: str) -> SessionResponse:
    session = _require_session(session_id)
    data = session.to_dict()
    data["turn_count"] = len(session.turns)
    return SessionResponse(**data)


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str) -> dict[str, str]:
    if not sessions.delete(session_id):
        raise HTTPException(404, "Session not found")
    return {"status": "deleted", "session_id": session_id}


@app.post("/api/sessions/{session_id}/turn", response_model=TurnResponse)
async def post_turn(session_id: str, audio: UploadFile = File(...)) -> TurnResponse:
    session = _require_session(session_id)
    if not pipeline.ready:
        raise HTTPException(503, pipeline.error or "Pipeline not ready")

    suffix = Path(audio.filename or "input.webm").suffix or ".wav"
    if suffix.lower() not in (".wav", ".webm", ".ogg", ".mp3", ".m4a"):
        suffix = ".wav"

    audio_dir = sessions.session_audio_dir(session_id)
    turn_index = len(session.turns) + 1
    user_name = f"turn_{turn_index:03d}_user{suffix}"
    reply_name = f"turn_{turn_index:03d}_reply.wav"
    user_path = audio_dir / user_name
    reply_path = audio_dir / reply_name

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        shutil.copyfileobj(audio.file, tmp)

    user_wav = audio_dir / f"turn_{turn_index:03d}_user.wav"
    try:
        shutil.copy2(tmp_path, user_path)
        prepare_wav_for_asr(tmp_path, user_wav)
        history = session.history_messages()
        context_messages = len(history)
        profile = pipeline.run_turn(user_wav, history=history, out_wav_path=reply_path)
    except RuntimeError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        logger.exception("Turn failed")
        raise HTTPException(500, f"Pipeline error: {e}") from e
    finally:
        tmp_path.unlink(missing_ok=True)

    turn = TurnRecord(
        turn_index=turn_index,
        created_at=_utc_now(),
        user_transcript=profile.transcript,
        assistant_reply=profile.reply_text,
        user_audio=user_wav.name,
        reply_audio=reply_name,
        timings={
            "asr_s": profile.asr_s,
            "llm_ttft_s": profile.llm_ttft_s,
            "llm_generation_s": profile.llm_generation_s,
            "tts_s": profile.tts_s,
        },
    )
    session.turns.append(turn)
    sessions.update(session)

    return TurnResponse(
        turn_index=turn_index,
        transcript=profile.transcript,
        reply_text=profile.reply_text,
        reply_audio_url=f"/api/sessions/{session_id}/audio/{reply_name}",
        user_audio_url=f"/api/sessions/{session_id}/audio/{user_wav.name}",
        timings=Timings(
            asr_s=profile.asr_s,
            llm_ttft_s=profile.llm_ttft_s,
            llm_generation_s=profile.llm_generation_s,
            tts_s=profile.tts_s,
        ),
        context_messages=context_messages,
        turn_count=len(session.turns),
    )


@app.get("/api/sessions/{session_id}/audio/{filename}")
def get_audio(session_id: str, filename: str) -> FileResponse:
    _require_session(session_id)
    if ".." in filename or "/" in filename:
        raise HTTPException(400, "Invalid filename")
    path = sessions.session_audio_dir(session_id) / filename
    if not path.is_file():
        raise HTTPException(404, "Audio not found")
    return FileResponse(path)


def _require_session(session_id: str) -> Session:
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(404, "Session not found")
    return session


if cfg.static_dir.is_dir():
    app.mount("/", StaticFiles(directory=str(cfg.static_dir), html=True), name="static")
