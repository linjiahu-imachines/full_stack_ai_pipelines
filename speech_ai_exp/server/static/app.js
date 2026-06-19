const chatLog = document.getElementById("chat-log");
const contextBanner = document.getElementById("context-banner");
const btnNewSession = document.getElementById("btn-new-session");
const btnHistory = document.getElementById("btn-history");
const btnCloseHistory = document.getElementById("btn-close-history");
const historyOverlay = document.getElementById("history-overlay");
const historyCurrent = document.getElementById("history-current");
const historyAll = document.getElementById("history-all");
const btnRecord = document.getElementById("btn-record");
const btnStop = document.getElementById("btn-stop");
const listenStatus = document.getElementById("listen-status");
const sessionLabel = document.getElementById("session-label");
const turnCountLabel = document.getElementById("turn-count-label");
const statusLabel = document.getElementById("status-label");
const busyEl = document.getElementById("busy");
const micBanner = document.getElementById("mic-banner");
const fileUpload = document.getElementById("file-upload");
const llmModelSelect = document.getElementById("llm-model-select");
const toolsEnabledSelect = document.getElementById("tools-enabled-select");

const LLM_MODEL_STORAGE_KEY = "chat_llm_model_id";
const TOOLS_ENABLED_STORAGE_KEY = "chat_tools_enabled";
let llmModels = [];

function isSecureMicContext() {
  if (window.isSecureContext) return true;
  const h = location.hostname;
  return h === "localhost" || h === "127.0.0.1" || h === "[::1]";
}

function micUnavailableMessage() {
  return (
    "Microphone is blocked: browsers only allow the mic on HTTPS or http://127.0.0.1 (not http://172.16.x.x). " +
    "Fix: on your PC run SSH tunnel, then open http://127.0.0.1:8000/ — " +
    "ssh -L 8000:127.0.0.1:8000 linhu@172.16.1.103 — " +
    "Or use Upload WAV below."
  );
}

async function getMicStream() {
  if (navigator.mediaDevices?.getUserMedia) {
    return navigator.mediaDevices.getUserMedia({ audio: true });
  }
  const legacy = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia;
  if (legacy) {
    return new Promise((resolve, reject) => {
      legacy.call(navigator, { audio: true }, resolve, reject);
    });
  }
  throw new Error(micUnavailableMessage());
}

function updateMicBanner() {
  if (!micBanner) return;
  if (isSecureMicContext() && navigator.mediaDevices?.getUserMedia) {
    micBanner.classList.add("hidden");
    micBanner.textContent = "";
    return;
  }
  micBanner.classList.remove("hidden");
  micBanner.innerHTML =
    "<strong>Microphone unavailable at this URL.</strong> " +
    "Use <code>http://127.0.0.1:8000/</code> via SSH tunnel " +
    "(<code>ssh -L 8000:127.0.0.1:8000 linhu@172.16.1.103</code>) " +
    "or <strong>Upload WAV</strong> to test without the mic.";
}

let sessionId = localStorage.getItem("chat_session_id") || null;

/** Voice-activity detection tuning (browser mic). */
const VAD = {
  SILENCE_MS: 1000,
  MIN_SPEECH_MS: 400,
  MAX_TURN_MS: 30000,
  TICK_MS: 50,
  NOISE_ADAPT: 0.92,
  SPEECH_MULT: 2.8,
  MIN_THRESHOLD: 0.012,
};

const autoVoice = {
  enabled: false,
  phase: "off", // off | armed | speaking | processing
  stream: null,
  audioContext: null,
  analyser: null,
  vadTimer: null,
  mediaRecorder: null,
  chunks: [],
  noiseFloor: 0.01,
  speechStartedAt: 0,
  recordStartedAt: 0,
  silenceStartedAt: null,
  finishing: false,
};

function rmsFromAnalyser(analyser) {
  const data = new Float32Array(analyser.fftSize);
  analyser.getFloatTimeDomainData(data);
  let sum = 0;
  for (let i = 0; i < data.length; i++) sum += data[i] * data[i];
  return Math.sqrt(sum / data.length);
}

function speechThreshold() {
  return Math.max(VAD.MIN_THRESHOLD, autoVoice.noiseFloor * VAD.SPEECH_MULT);
}

function updateListenUi() {
  if (!listenStatus) return;
  const { phase, enabled } = autoVoice;
  if (!enabled || phase === "off") {
    listenStatus.classList.add("hidden");
    listenStatus.classList.remove("speaking", "processing");
    listenStatus.textContent = "";
    btnRecord.textContent = "Start listening";
    btnRecord.classList.remove("listening", "recording");
    btnStop.textContent = "Stop listening";
    btnStop.disabled = true;
    return;
  }

  listenStatus.classList.remove("hidden");
  listenStatus.classList.remove("speaking", "processing");
  btnRecord.disabled = true;
  btnRecord.textContent = "Listening…";
  btnRecord.classList.add("listening");
  btnRecord.classList.remove("recording");
  btnStop.disabled = phase === "processing";
  btnStop.textContent = phase === "speaking" ? "Send now" : "Stop listening";

  if (phase === "armed") {
    listenStatus.textContent = "Waiting for you to speak… (pause ~1s when done)";
  } else if (phase === "speaking") {
    listenStatus.classList.add("speaking");
    listenStatus.textContent = "Hearing you — pause when finished, or click Send now";
    btnRecord.classList.remove("listening");
    btnRecord.classList.add("recording");
  } else if (phase === "processing") {
    listenStatus.classList.add("processing");
    listenStatus.textContent = "Processing your message…";
    btnRecord.classList.remove("listening", "recording");
  }
}

function clearVadTimer() {
  if (autoVoice.vadTimer) {
    clearInterval(autoVoice.vadTimer);
    autoVoice.vadTimer = null;
  }
}

function releaseAutoVoiceStream() {
  clearVadTimer();
  if (autoVoice.mediaRecorder && autoVoice.mediaRecorder.state !== "inactive") {
    try {
      autoVoice.mediaRecorder.stop();
    } catch {
      /* ignore */
    }
  }
  autoVoice.mediaRecorder = null;
  autoVoice.chunks = [];
  if (autoVoice.stream) {
    autoVoice.stream.getTracks().forEach((t) => t.stop());
    autoVoice.stream = null;
  }
  if (autoVoice.audioContext) {
    autoVoice.audioContext.close().catch(() => {});
    autoVoice.audioContext = null;
  }
  autoVoice.analyser = null;
  autoVoice.finishing = false;
}

function stopAutoListen() {
  autoVoice.enabled = false;
  autoVoice.phase = "off";
  releaseAutoVoiceStream();
  updateListenUi();
  if (!busyEl.classList.contains("hidden")) return;
  btnRecord.disabled = false;
}

function beginAutoRecording() {
  if (!autoVoice.enabled || autoVoice.phase !== "armed" || !autoVoice.stream) return;
  autoVoice.chunks = [];
  autoVoice.finishing = false;
  autoVoice.silenceStartedAt = null;
  autoVoice.speechStartedAt = Date.now();
  autoVoice.recordStartedAt = autoVoice.speechStartedAt;
  autoVoice.mediaRecorder = new MediaRecorder(autoVoice.stream);
  autoVoice.mediaRecorder.ondataavailable = (e) => {
    if (e.data.size > 0) autoVoice.chunks.push(e.data);
  };
  autoVoice.mediaRecorder.start(VAD.TICK_MS);
  autoVoice.phase = "speaking";
  updateListenUi();
}

async function finishAutoRecording({ force = false } = {}) {
  if (!autoVoice.enabled || autoVoice.phase !== "speaking" || autoVoice.finishing) return;
  const spokeMs = Date.now() - autoVoice.speechStartedAt;
  if (!force && spokeMs < VAD.MIN_SPEECH_MS) return;

  autoVoice.finishing = true;
  const recorder = autoVoice.mediaRecorder;
  if (!recorder || recorder.state === "inactive") {
    autoVoice.finishing = false;
    autoVoice.phase = "armed";
    autoVoice.silenceStartedAt = null;
    updateListenUi();
    return;
  }

  recorder.onstop = async () => {
    const mime = recorder.mimeType || "audio/webm";
    autoVoice.mediaRecorder = null;
    const blob = new Blob(autoVoice.chunks, { type: mime });
    autoVoice.chunks = [];
    autoVoice.finishing = false;
    autoVoice.silenceStartedAt = null;

    if (blob.size < 256) {
      autoVoice.phase = autoVoice.enabled ? "armed" : "off";
      updateListenUi();
      return;
    }

    autoVoice.phase = "processing";
    updateListenUi();
    await sendTurn(blob);
  };
  recorder.stop();
}

function vadTick() {
  if (!autoVoice.enabled || !autoVoice.analyser) return;
  if (autoVoice.phase === "processing" || autoVoice.finishing) return;

  const rms = rmsFromAnalyser(autoVoice.analyser);
  const threshold = speechThreshold();
  const now = Date.now();
  const isSpeech = rms > threshold;

  if (autoVoice.phase === "armed") {
    autoVoice.noiseFloor =
      autoVoice.noiseFloor * VAD.NOISE_ADAPT + rms * (1 - VAD.NOISE_ADAPT);
    if (isSpeech) beginAutoRecording();
    return;
  }

  if (autoVoice.phase !== "speaking") return;

  if (isSpeech) {
    autoVoice.silenceStartedAt = null;
    return;
  }

  if (!autoVoice.silenceStartedAt) autoVoice.silenceStartedAt = now;

  const silentMs = now - autoVoice.silenceStartedAt;
  const spokeMs = now - autoVoice.speechStartedAt;
  const longEnough = spokeMs >= VAD.MIN_SPEECH_MS;
  const pausedEnough = silentMs >= VAD.SILENCE_MS;
  const hitMax = now - autoVoice.recordStartedAt >= VAD.MAX_TURN_MS;

  if ((longEnough && pausedEnough) || hitMax) {
    finishAutoRecording({ force: hitMax });
  }
}

function resumeAutoListenAfterTurn() {
  if (!autoVoice.enabled || !autoVoice.stream) {
    stopAutoListen();
    return;
  }
  autoVoice.phase = "armed";
  autoVoice.silenceStartedAt = null;
  autoVoice.finishing = false;
  updateListenUi();
}

async function startAutoListen() {
  if (autoVoice.enabled) return;
  if (!sessionId) await createSession(true);
  try {
    const stream = await getMicStream();
    const audioContext = new AudioContext();
    await audioContext.resume();
    const source = audioContext.createMediaStreamSource(stream);
    const analyser = audioContext.createAnalyser();
    analyser.fftSize = 2048;
    source.connect(analyser);

    autoVoice.stream = stream;
    autoVoice.audioContext = audioContext;
    autoVoice.analyser = analyser;
    autoVoice.enabled = true;
    autoVoice.phase = "armed";
    autoVoice.noiseFloor = 0.01;
    autoVoice.silenceStartedAt = null;
    autoVoice.finishing = false;

    clearVadTimer();
    autoVoice.vadTimer = setInterval(vadTick, VAD.TICK_MS);

    updateListenUi();
  } catch (e) {
    stopAutoListen();
    alert("Microphone error: " + e.message);
  }
}

async function checkHealth() {
  try {
    const r = await fetch("/health");
    const j = await r.json();
    statusLabel.textContent = j.pipeline_ready ? "Pipeline ready" : "Loading models…";
    if (j.pipeline_ready && Array.isArray(j.llm_models) && j.llm_models.length && !llmModels.length) {
      llmModels = j.llm_models;
      populateLlmModelSelect(j.llm_default);
      if (sessionId) {
        const session = await fetchSession(sessionId);
        if (session?.turns?.length) renderSession(session);
      }
    }
  } catch {
    statusLabel.textContent = "Offline";
  }
}

async function loadLlmModels() {
  try {
    const r = await fetch("/api/llm-models");
    if (!r.ok) return;
    llmModels = await r.json();
    const defaultId = llmModels.find((m) => m.is_default)?.id || llmModels[0]?.id;
    populateLlmModelSelect(defaultId);
    if (sessionId) {
      const session = await fetchSession(sessionId);
      if (session?.turns?.length) renderSession(session);
    }
  } catch (e) {
    console.warn("Failed to load LLM models:", e);
  }
}

function llmModelLabelForId(modelId) {
  if (!modelId || !llmModels.length) return "";
  const hit = llmModels.find((m) => m.id === modelId);
  return hit?.label || "";
}

/** Prefer current registry label; fall back to label stored on the turn. */
function llmModelLabelForTurn(turn) {
  const agent = turn?.agent || {};
  const fromRegistry = llmModelLabelForId(agent.llm_model);
  if (fromRegistry) return fromRegistry;
  return agent.llm_model_label || "";
}

function llmModelLabelFromResponse(j) {
  const fromRegistry = llmModelLabelForId(j?.llm_model);
  if (fromRegistry) return fromRegistry;
  return j?.llm_model_label || "";
}

function populateLlmModelSelect(defaultId) {
  if (!llmModelSelect || !llmModels.length) return;
  const saved = localStorage.getItem(LLM_MODEL_STORAGE_KEY);
  const selected = saved && llmModels.some((m) => m.id === saved) ? saved : defaultId;
  llmModelSelect.innerHTML = "";
  for (const m of llmModels) {
    const opt = document.createElement("option");
    opt.value = m.id;
    opt.textContent = m.label;
    if (m.id === selected) opt.selected = true;
    llmModelSelect.appendChild(opt);
  }
  if (selected) localStorage.setItem(LLM_MODEL_STORAGE_KEY, selected);
}

function selectedLlmModelId() {
  if (!llmModelSelect) return "";
  const id = llmModelSelect.value;
  if (id) localStorage.setItem(LLM_MODEL_STORAGE_KEY, id);
  return id;
}

function isRemoteLlmSelected() {
  const id = selectedLlmModelId();
  const m = llmModels.find((x) => x.id === id);
  return m?.backend === "remote";
}

function setBusyMessage() {
  if (!busyEl) return;
  busyEl.textContent = isRemoteLlmSelected()
    ? "Processing turn… remote IMI-RISCV simulator LLM can take many minutes; please wait."
    : "Processing turn…";
}

if (llmModelSelect) {
  llmModelSelect.addEventListener("change", () => {
    selectedLlmModelId();
  });
}

function populateToolsEnabledSelect() {
  if (!toolsEnabledSelect) return;
  const saved = localStorage.getItem(TOOLS_ENABLED_STORAGE_KEY);
  toolsEnabledSelect.value = saved === "false" ? "false" : "true";
}

function selectedToolsEnabled() {
  if (!toolsEnabledSelect) return true;
  const on = toolsEnabledSelect.value !== "false";
  localStorage.setItem(TOOLS_ENABLED_STORAGE_KEY, on ? "true" : "false");
  return on;
}

if (toolsEnabledSelect) {
  toolsEnabledSelect.addEventListener("change", () => {
    selectedToolsEnabled();
  });
  populateToolsEnabledSelect();
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function formatTiming(timings) {
  if (!timings) return "";
  return `ASR ${Number(timings.asr_s).toFixed(2)}s · LLM ${Number(timings.llm_generation_s).toFixed(2)}s · TTS ${Number(timings.tts_s).toFixed(2)}s`;
}

function formatLlmUsage(usage) {
  if (!usage) return "";
  const inTok = Math.round(Number(usage.prompt_tokens_est) || 0);
  const outTok = Math.round(Number(usage.output_tokens_est) || 0);
  const calls = Number(usage.llm_calls) || 1;
  const callNote = calls > 1 ? ` · ${calls} LLM calls` : "";
  return (
    `LLM in ${Number(usage.prompt_chars).toLocaleString()} chars (~${inTok.toLocaleString()} tok)` +
    ` · out ${Number(usage.output_chars).toLocaleString()} chars (~${outTok.toLocaleString()} tok)` +
    callNote
  );
}

function formatWhen(iso) {
  if (!iso) return "";
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

function playReplyAudio(url) {
  if (!url) return;
  const audio = new Audio(url);
  audio.play().catch((err) => console.warn("Auto-play blocked:", err));
  return audio;
}

function renderTurn(turn, sid) {
  const group = document.createElement("div");
  group.className = "turn-group";
  const userMsg = document.createElement("div");
  userMsg.className = "msg user";
  userMsg.innerHTML = `
    <div class="role">You</div>
    <div class="text">${escapeHtml(turn.user_transcript || "(no transcript)")}</div>
    <audio controls src="/api/sessions/${sid}/audio/${encodeURIComponent(turn.user_audio)}"></audio>
  `;
  const asstMsg = document.createElement("div");
  asstMsg.className = "msg assistant";
  asstMsg.innerHTML = `
    <div class="role">Assistant</div>
    <div class="text">${escapeHtml(turn.assistant_reply || "")}</div>
    <audio controls src="/api/sessions/${sid}/audio/${encodeURIComponent(turn.reply_audio)}"></audio>
  `;
  group.appendChild(document.createElement("div")).className = "turn-header";
  group.querySelector(".turn-header").textContent = `Turn ${turn.turn_index}`;
  group.appendChild(userMsg);
  group.appendChild(asstMsg);
  return group;
}

function updateContextBanner(turnCount) {
  if (!sessionId || turnCount === 0) {
    contextBanner.classList.add("hidden");
    return;
  }
  contextBanner.classList.remove("hidden");
  const prior = Math.max(0, turnCount - 1);
  contextBanner.textContent = prior === 0
    ? "First turn — no prior context yet."
    : `Multi-turn: ${prior} prior turn(s) sent to the LLM.`;
}

function updateSessionLabels(turnCount) {
  if (sessionId) {
    sessionLabel.textContent = `Active session · ${sessionId.slice(0, 8)}…`;
    turnCountLabel.textContent = turnCount ? `${turnCount} turn(s)` : "0 turns";
  } else {
    sessionLabel.textContent = "No active session";
    turnCountLabel.textContent = "";
  }
}

function renderSession(session) {
  const turns = session.turns || [];
  const sid = session.session_id;
  chatLog.innerHTML = "";
  if (turns.length === 0) {
    chatLog.innerHTML = '<p class="empty-hint">Session ready. Click <strong>Start listening</strong> and speak.</p>';
  } else {
    for (const t of turns) chatLog.appendChild(renderTurn(t, sid));
    chatLog.scrollTop = chatLog.scrollHeight;
  }
  const count = session.turn_count ?? turns.length;
  updateSessionLabels(count);
  updateContextBanner(count);
}

async function fetchSession(id = sessionId) {
  const sid = id || sessionId;
  if (!sid) return null;
  const r = await fetch(`/api/sessions/${sid}`);
  if (!r.ok) return null;
  return r.json();
}

async function fetchAllSessions() {
  const r = await fetch("/api/sessions");
  return r.ok ? r.json() : [];
}

async function createSession(skipConfirm = false) {
  if (
    !skipConfirm &&
    sessionId &&
    (await fetchSession())?.turns?.length > 0 &&
    !confirm("Start a new conversation? Current session stays saved on the server.")
  ) {
    return null;
  }
  if (autoVoice.enabled) stopAutoListen();
  const r = await fetch("/api/sessions", { method: "POST" });
  if (!r.ok) throw new Error(await r.text());
  const j = await r.json();
  sessionId = j.session_id;
  localStorage.setItem("chat_session_id", sessionId);
  renderSession({ session_id: sessionId, turns: [], turn_count: 0 });
  return sessionId;
}

async function loadSession(id = sessionId) {
  if (!sessionId && !id) {
    updateSessionLabels(0);
    updateContextBanner(0);
    return;
  }
  if (id && id !== sessionId) {
    sessionId = id;
    localStorage.setItem("chat_session_id", sessionId);
  }
  const session = await fetchSession(sessionId);
  if (!session) {
    sessionId = null;
    localStorage.removeItem("chat_session_id");
    chatLog.innerHTML = '<p class="empty-hint">Session not found. Start a <strong>New conversation</strong>.</p>';
    updateSessionLabels(0);
    updateContextBanner(0);
    return;
  }
  renderSession(session);
}

function closeHistory() {
  historyOverlay.classList.add("hidden");
  document.body.style.overflow = "";
}

async function refreshHistoryPanel() {
  const session = sessionId ? await fetchSession() : null;
  historyCurrent.innerHTML = "";
  if (!session?.turns?.length) {
    historyCurrent.innerHTML = '<p class="history-empty">No turns in this session yet.</p>';
  } else {
    for (const t of session.turns) {
      const row = document.createElement("div");
      row.className = "history-turn";
      row.innerHTML = `<strong>Turn ${t.turn_index}</strong><br/>You: ${escapeHtml(t.user_transcript || "—")}<br/>Assistant: ${escapeHtml(t.assistant_reply || "—")}`;
      historyCurrent.appendChild(row);
    }
  }
  historyAll.innerHTML = '<p class="history-empty">Loading…</p>';
  const list = await fetchAllSessions();
  historyAll.innerHTML = "";
  if (!list.length) {
    historyAll.innerHTML = '<p class="history-empty">No saved sessions yet.</p>';
    return;
  }
  for (const s of list) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "history-item" + (s.session_id === sessionId ? " active" : "");
    btn.innerHTML = `<div>${escapeHtml(s.session_id.slice(0, 12))}… · ${s.turn_count} turn(s)</div><div>${escapeHtml(formatWhen(s.updated_at))}</div>`;
    btn.addEventListener("click", async () => {
      closeHistory();
      await loadSession(s.session_id);
    });
    historyAll.appendChild(btn);
  }
}

btnNewSession.addEventListener("click", () => createSession(false));
btnHistory.addEventListener("click", () => {
  historyOverlay.classList.remove("hidden");
  document.body.style.overflow = "hidden";
  refreshHistoryPanel();
});
btnCloseHistory.addEventListener("click", closeHistory);
historyOverlay.addEventListener("click", (e) => { if (e.target === historyOverlay) closeHistory(); });

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    tab.classList.add("active");
    const name = tab.dataset.tab;
    historyCurrent.classList.toggle("hidden", name !== "current");
    historyAll.classList.toggle("hidden", name !== "all");
    if (name === "all") refreshHistoryPanel();
  });
});

async function sendTurn(blob) {
  if (!sessionId) await createSession(true);
  setBusyMessage();
  busyEl.classList.remove("hidden");
  btnRecord.disabled = true;
  btnStop.disabled = true;
  const form = new FormData();
  form.append("audio", blob, "utterance.webm");
  const modelId = selectedLlmModelId();
  if (modelId) form.append("llm_model", modelId);
  form.append("tools_enabled", selectedToolsEnabled() ? "true" : "false");
  try {
    const r = await fetch(`/api/sessions/${sessionId}/turn`, { method: "POST", body: form });
    if (!r.ok) throw new Error(await r.text() || r.statusText);
    const j = await r.json();
    await loadSession();
    const groups = chatLog.querySelectorAll(".turn-group");
    const el = groups[groups.length - 1]?.querySelector(".msg.assistant audio");
    if (el) el.play().catch(() => playReplyAudio(j.reply_audio_url));
    else playReplyAudio(j.reply_audio_url);
    updateContextBanner(j.turn_count ?? 0);
  } catch (e) {
    const errBox = document.createElement("div");
    errBox.className = "msg assistant";
    errBox.innerHTML = `<div class="role">Error</div><div class="text">${escapeHtml(e.message)}</div>`;
    chatLog.appendChild(errBox);
  } finally {
    busyEl.classList.add("hidden");
    if (autoVoice.enabled) {
      resumeAutoListenAfterTurn();
    } else {
      btnRecord.disabled = false;
      btnStop.disabled = true;
      btnRecord.classList.remove("recording", "listening");
      updateListenUi();
    }
  }
}

btnRecord.addEventListener("click", async () => {
  if (autoVoice.enabled) return;
  await startAutoListen();
});

if (fileUpload) {
  fileUpload.addEventListener("change", async () => {
    const file = fileUpload.files?.[0];
    fileUpload.value = "";
    if (!file) return;
    if (autoVoice.enabled) stopAutoListen();
    await sendTurn(file);
  });
}

btnStop.addEventListener("click", async () => {
  if (!autoVoice.enabled) return;
  if (autoVoice.phase === "speaking") {
    await finishAutoRecording({ force: true });
    return;
  }
  if (autoVoice.phase === "armed") {
    stopAutoListen();
  }
});

(async () => {
  updateMicBanner();
  await checkHealth();
  await loadLlmModels();
  await loadSession();
})();
