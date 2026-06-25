from __future__ import annotations

import logging
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
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
_server_root = Path(__file__).resolve().parents[1]
pipeline = PipelineService(cfg.staged_config, server_root=_server_root)


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


class LlmUsage(BaseModel):
    prompt_chars: int = 0
    prompt_tokens_est: float = 0
    system_chars: int = 0
    messages_chars: int = 0
    output_chars: int = 0
    output_tokens_est: float = 0
    llm_calls: int = 1


class TurnResponse(BaseModel):
    turn_index: int
    transcript: str
    reply_text: str
    reply_audio_url: str = ""
    user_audio_url: str = ""
    input_mode: str = "voice"
    timings: Timings
    context_messages: int
    turn_count: int
    rag_sources: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    agent_enabled: bool = False
    llm_model: str = ""
    llm_model_label: str = ""
    tools_enabled: bool = True
    llm_usage: LlmUsage | None = None


class TextTurnRequest(BaseModel):
    text: str
    llm_model: str | None = None
    tools_enabled: bool = True
    speak_reply: bool = False


class LlmModelOption(BaseModel):
    id: str
    label: str
    backend: str
    loaded: bool = False
    is_default: bool = False


class SessionResponse(BaseModel):
    session_id: str
    created_at: str
    updated_at: str
    turn_count: int
    turns: list[dict[str, Any]]


@app.get("/health")
def health() -> dict[str, Any]:
    default_llm = pipeline.llm_registry.default_id if pipeline.llm_registry else None
    return {
        "status": "ok" if pipeline.ready else "loading",
        "pipeline_ready": pipeline.ready,
        "pipeline_error": pipeline.error,
        "config": str(cfg.staged_config),
        "agent": pipeline.agent_info if pipeline.ready else {"enabled": False},
        "llm_models": pipeline.list_llm_models() if pipeline.ready else [],
        "llm_default": default_llm,
    }


@app.get("/api/llm-models", response_model=list[LlmModelOption])
def list_llm_models() -> list[LlmModelOption]:
    if not pipeline.ready:
        raise HTTPException(503, pipeline.error or "Pipeline not ready")
    default_id = pipeline.llm_registry.default_id if pipeline.llm_registry else ""
    return [
        LlmModelOption(
            id=m["id"],
            label=m["label"],
            backend=m["backend"],
            loaded=m.get("loaded", False),
            is_default=m["id"] == default_id,
        )
        for m in pipeline.list_llm_models()
    ]


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


def _parse_form_bool(value: str | None, *, default: bool = True) -> bool:
    if value is None or value == "":
        return default
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _llm_usage_from_profile(profile, agent_meta: dict, llm_meta: dict) -> LlmUsage | None:
    usage_raw = dict(llm_meta.get("usage") or agent_meta.get("llm_usage") or {})
    if not usage_raw:
        return None
    return LlmUsage(
        prompt_chars=int(usage_raw.get("prompt_chars", 0)),
        prompt_tokens_est=float(usage_raw.get("prompt_tokens_est", 0)),
        system_chars=int(usage_raw.get("system_chars", 0)),
        messages_chars=int(usage_raw.get("messages_chars", 0)),
        output_chars=int(usage_raw.get("output_chars", 0)),
        output_tokens_est=float(usage_raw.get("output_tokens_est", 0)),
        llm_calls=int(usage_raw.get("llm_calls", 1)),
    )


def _resolve_model_label(llm_model_id: str, fallback: str = "") -> str:
    for m in pipeline.list_llm_models():
        if m["id"] == llm_model_id:
            return str(m["label"])
    return fallback or llm_model_id


def _save_turn_and_respond(
    *,
    session: Session,
    session_id: str,
    turn_index: int,
    profile,
    context_messages: int,
    use_tools: bool,
    llm_model_id: str,
    model_label: str,
    input_mode: str,
    user_audio: str,
    reply_audio: str,
) -> TurnResponse:
    agent_meta = dict(profile.meta.get("agent") or {})
    llm_meta = dict(profile.meta.get("llm") or {})
    llm_usage = _llm_usage_from_profile(profile, agent_meta, llm_meta)
    usage_raw = dict(llm_meta.get("usage") or agent_meta.get("llm_usage") or {})
    turn = TurnRecord(
        turn_index=turn_index,
        created_at=_utc_now(),
        user_transcript=profile.transcript,
        assistant_reply=profile.reply_text,
        user_audio=user_audio,
        reply_audio=reply_audio,
        input_mode=input_mode,
        timings={
            "asr_s": profile.asr_s,
            "llm_ttft_s": profile.llm_ttft_s,
            "llm_generation_s": profile.llm_generation_s,
            "tts_s": profile.tts_s,
        },
        agent={
            **agent_meta,
            "llm_model": llm_model_id,
            "llm_model_label": model_label,
            "tools_enabled": use_tools,
            "llm_usage": usage_raw,
            "input_mode": input_mode,
        },
    )
    session.turns.append(turn)
    sessions.update(session)
    reply_audio_url = (
        f"/api/sessions/{session_id}/audio/{reply_audio}" if reply_audio else ""
    )
    user_audio_url = (
        f"/api/sessions/{session_id}/audio/{user_audio}" if user_audio else ""
    )
    return TurnResponse(
        turn_index=turn_index,
        transcript=profile.transcript,
        reply_text=profile.reply_text,
        reply_audio_url=reply_audio_url,
        user_audio_url=user_audio_url,
        input_mode=input_mode,
        timings=Timings(
            asr_s=profile.asr_s,
            llm_ttft_s=profile.llm_ttft_s,
            llm_generation_s=profile.llm_generation_s,
            tts_s=profile.tts_s,
        ),
        context_messages=context_messages,
        turn_count=len(session.turns),
        rag_sources=list(agent_meta.get("rag_sources") or []),
        tool_calls=list(agent_meta.get("tool_calls") or []),
        agent_enabled=pipeline.agent_enabled,
        llm_model=llm_model_id,
        llm_model_label=model_label,
        tools_enabled=use_tools,
        llm_usage=llm_usage,
    )


@app.post("/api/sessions/{session_id}/turn", response_model=TurnResponse)
async def post_turn(
    session_id: str,
    audio: UploadFile = File(...),
    llm_model: str | None = Form(default=None),
    tools_enabled: str | None = Form(default="true"),
) -> TurnResponse:
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
        use_tools = _parse_form_bool(tools_enabled, default=True)
        profile = pipeline.run_turn(
            user_wav,
            history=history,
            out_wav_path=reply_path,
            session_memory=session.agent_memory,
            llm_model=llm_model,
            tools_enabled=use_tools,
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except RuntimeError as e:
        msg = str(e)
        if "timed out" in msg.lower() or "unreachable" in msg.lower():
            raise HTTPException(504, msg) from e
        raise HTTPException(400, msg) from e
    except TimeoutError as e:
        raise HTTPException(
            504,
            "LLM request timed out. Try the local Qwen model or increase REMOTE_LLM_TIMEOUT_SEC.",
        ) from e
    except Exception as e:
        logger.exception("Turn failed")
        raise HTTPException(500, f"Pipeline error: {e}") from e
    finally:
        tmp_path.unlink(missing_ok=True)

    llm_meta = dict(profile.meta.get("llm") or {})
    llm_model_id = str(llm_meta.get("model_id") or llm_model or "")
    model_label = _resolve_model_label(llm_model_id, llm_model_id)
    return _save_turn_and_respond(
        session=session,
        session_id=session_id,
        turn_index=turn_index,
        profile=profile,
        context_messages=context_messages,
        use_tools=use_tools,
        llm_model_id=llm_model_id,
        model_label=model_label,
        input_mode="voice",
        user_audio=user_wav.name,
        reply_audio=reply_name,
    )


@app.post("/api/sessions/{session_id}/turn/text", response_model=TurnResponse)
async def post_text_turn(session_id: str, body: TextTurnRequest) -> TurnResponse:
    session = _require_session(session_id)
    if not pipeline.ready:
        raise HTTPException(503, pipeline.error or "Pipeline not ready")

    text = body.text.strip()
    if not text:
        raise HTTPException(400, "Message text is required")

    audio_dir = sessions.session_audio_dir(session_id)
    turn_index = len(session.turns) + 1
    reply_name = f"turn_{turn_index:03d}_reply.wav"
    reply_path = audio_dir / reply_name if body.speak_reply else None
    history = session.history_messages()
    context_messages = len(history)

    try:
        profile = pipeline.run_text_turn(
            text,
            history=history,
            out_wav_path=reply_path,
            session_memory=session.agent_memory,
            llm_model=body.llm_model,
            tools_enabled=body.tools_enabled,
            speak_reply=body.speak_reply,
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except RuntimeError as e:
        msg = str(e)
        if "timed out" in msg.lower() or "unreachable" in msg.lower():
            raise HTTPException(504, msg) from e
        raise HTTPException(400, msg) from e
    except TimeoutError as e:
        raise HTTPException(
            504,
            "LLM request timed out. Try the local Qwen model or increase REMOTE_LLM_TIMEOUT_SEC.",
        ) from e
    except Exception as e:
        logger.exception("Text turn failed")
        raise HTTPException(500, f"Pipeline error: {e}") from e

    llm_meta = dict(profile.meta.get("llm") or {})
    llm_model_id = str(llm_meta.get("model_id") or body.llm_model or "")
    model_label = _resolve_model_label(llm_model_id, llm_model_id)
    reply_audio = reply_name if body.speak_reply and profile.output_wav_path else ""
    return _save_turn_and_respond(
        session=session,
        session_id=session_id,
        turn_index=turn_index,
        profile=profile,
        context_messages=context_messages,
        use_tools=body.tools_enabled and pipeline.agent_enabled,
        llm_model_id=llm_model_id,
        model_label=model_label,
        input_mode="text",
        user_audio="",
        reply_audio=reply_audio,
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
