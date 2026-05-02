"""FastAPI backend for the Insurance Claim Live Agent Team UI.

The browser sends claimant turns here. This backend calls Gemini structured
output for language-heavy extraction/classification, then runs the existing
deterministic policy gates from policies.py.
"""

from __future__ import annotations

import asyncio
import base64
import os
import re
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

APP_DIR = Path(__file__).resolve().parents[1]
DEMO_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from policies import (  # noqa: E402
    apply_coverage_and_evidence_rules,
    build_claim_intake_packet,
    fraud_signal_and_safety_gate,
    generate_document_checklist,
    validate_required_claim_fields,
)
from schemas import ClaimClassification, ClaimNarrative  # noqa: E402

MODEL = os.getenv("FNOL_GEMINI_MODEL", "gemini-3-flash-preview")
LIVE_MODEL = os.getenv("FNOL_GEMINI_LIVE_MODEL", "gemini-3.1-flash-live-preview")
GENAI_CLIENT = None


class MessageRequest(BaseModel):
    session_id: str
    text: str


class SessionResponse(BaseModel):
    session_id: str
    model: str
    has_api_key: bool
    state: dict[str, Any]


class AgentReply(BaseModel):
    response_text: str = Field(
        description=(
            "Natural claimant-facing response. Ask the next needed question or "
            "confirm the handoff status. Do not promise coverage, payment, or liability."
        )
    )


@dataclass
class IntakeSession:
    session_id: str
    transcript: list[dict[str, str]] = field(default_factory=list)
    normalized_claim: dict[str, Any] | None = None
    classification: dict[str, Any] | None = None
    route: str = "needs_docs"


sessions: dict[str, IntakeSession] = {}

app = FastAPI(title="Insurance Claim Live Agent Team API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _load_dotenv() -> None:
    env_path = APP_DIR / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()


def _has_api_key() -> bool:
    return bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))


def _client():
    global GENAI_CLIENT
    if not _has_api_key():
        raise HTTPException(
            status_code=503,
            detail=(
                "Missing GOOGLE_API_KEY. Add it to "
                f"{APP_DIR / '.env'} and restart the live intake backend."
            ),
        )
    if os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]
    try:
        from google import genai
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail="Missing google-genai package. Run pip install -r requirements.txt.",
        ) from exc
    if GENAI_CLIENT is None:
        GENAI_CLIENT = genai.Client()
    return GENAI_CLIENT


def _blank_claim() -> dict[str, Any]:
    return {
        "policyholder_name": "not specified",
        "policy_number": "not specified",
        "contact_method": "not specified",
        "date_of_loss": "not specified",
        "reported_date": "not specified",
        "loss_location": "not specified",
        "loss_description": "not specified",
        "estimated_loss_usd": None,
        "injuries_or_safety_concerns": [],
        "parties_involved": [],
        "evidence_available": [],
        "documents_mentioned": [],
        "missing_or_uncertain_facts": [],
        "raw_narrative_summary": "not specified",
        "assumptions": [],
    }


def _claim_from_session(session: IntakeSession) -> dict[str, Any]:
    return session.normalized_claim or _blank_claim()


def _claimant_text(session: IntakeSession) -> str:
    return "\n".join(
        turn["text"] for turn in session.transcript if turn["speaker"] == "Claimant"
    )


def _generate_structured(prompt: str, schema: type[BaseModel]) -> dict[str, Any]:
    try:
        response = _client().models.generate_content(
            model=MODEL,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": schema.model_json_schema(),
            },
        )
        return schema.model_validate_json(response.text).model_dump(exclude_none=True)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Gemini request failed: {exc}") from exc


async def _generate_structured_async(prompt: str, schema: type[BaseModel]) -> dict[str, Any]:
    return await asyncio.to_thread(_generate_structured, prompt, schema)


def _transcribe_audio(audio_bytes: bytes, mime_type: str) -> str:
    try:
        from google.genai import types
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail="Missing google-genai package. Run pip install -r requirements.txt.",
        ) from exc

    try:
        response = _client().models.generate_content(
            model=MODEL,
            contents=[
                types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
                (
                    "Transcribe this claimant audio as plain text only. "
                    "Preserve names, phone numbers, policy numbers, locations, dates, "
                    "injuries, documents, and evidence exactly when audible."
                ),
            ],
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Gemini audio transcription failed: {exc}") from exc
    text = (response.text or "").strip()
    if not text:
        raise HTTPException(status_code=422, detail="No speech was transcribed from the audio.")
    return text


def _normalize_claim(session: IntakeSession) -> dict[str, Any]:
    current = _claim_from_session(session)
    prompt = f"""
You are an insurance FNOL intake extraction service.

Use the full claimant transcript to produce one complete ClaimNarrative.
Preserve known facts from the current claim state unless the transcript corrects them.
Do not invent policy numbers, contacts, dates, locations, evidence, or dollar amounts.
Use "not specified" for unknown string fields.

Current claim state:
{current}

Claimant transcript:
{_claimant_text(session)}
"""
    return _generate_structured(prompt, ClaimNarrative)


def _classify_claim(
    claim: dict[str, Any],
    validation: dict[str, Any],
) -> dict[str, Any]:
    prompt = f"""
Classify this insurance FNOL intake for operational routing.
Return a ClaimClassification only. Do not decide coverage or payment.

Normalized claim:
{claim}

Field validation:
{validation}

Severity rubric:
- low: complete, low-dollar, no injury/safety issue, routine documentation.
- medium: missing documents or moderate complexity.
- high: high estimated loss, unclear liability, missing core facts, or specialized handling likely.
- urgent: injury, unsafe living condition, emergency medical/safety concern, or time-sensitive mitigation.
"""
    return _generate_structured(prompt, ClaimClassification)


def _status(value: Any, urgent: bool = False) -> str:
    text = str(value or "").strip().lower()
    if urgent:
        return "urgent"
    if text in {"", "unknown", "not specified", "unspecified", "n/a", "none", "not provided"}:
        return "missing"
    return "complete"


def _without_negated_safety_mentions(text: str) -> str:
    cleaned = str(text or "")
    for pattern in [
        r"\b(?:no|not|none|without|denies|denied)\s+(?:one\s+)?(?:was\s+)?(?:injur\w*|hurt|pain|medical attention|ambulance|hospital|unsafe|hazard\w*|danger)\b",
        r"\b(?:injur\w*|hurt|pain|medical attention|ambulance|hospital|unsafe|hazard\w*|danger)\s+(?:was|were|is|are)?\s*(?:reported\s+)?(?:no|none|not reported|denied)\b",
    ]:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
    return cleaned


def _has_negated_safety_mention(text: str) -> bool:
    return _without_negated_safety_mentions(text) != str(text or "")


def _positive_safety_items(items: list[str]) -> list[str]:
    patterns = [
        r"\binjur",
        r"\bhurt\b",
        r"\bneck pain\b",
        r"\bhospital\b",
        r"\burgent care\b",
        r"\bambulance\b",
        r"\bunsafe\b",
        r"\bhazard",
        r"\bdanger\b",
    ]
    return [
        item
        for item in items
        if any(re.search(pattern, _without_negated_safety_mentions(item), flags=re.IGNORECASE) for pattern in patterns)
    ]


def _join(items: list[str], fallback: str) -> str:
    return ", ".join(items) if items else fallback


def _field(label: str, value: Any, source: str = "Gemini extraction", urgent: bool = False) -> dict[str, str]:
    status = _status(value, urgent=urgent)
    display = value if status != "missing" else f"Missing: {label.lower()}"
    return {
        "label": label,
        "value": str(display),
        "status": status,
        "source": "-" if status == "missing" else source,
    }


def _events(
    session: IntakeSession,
    validation: dict[str, Any],
    coverage: dict[str, Any],
    fraud_gate: dict[str, Any],
) -> list[dict[str, str]]:
    events: list[dict[str, str]] = [
        {
            "tone": "success",
            "title": "Gemini extraction complete",
            "detail": f"Updated structured claim facts using {MODEL}.",
            "rule": "LLM-001",
        }
    ]
    if validation.get("missing_fields"):
        events.append(
            {
                "tone": "warning",
                "title": "Missing intake facts",
                "detail": ", ".join(validation["missing_fields"]),
                "rule": "INTAKE-001",
            }
        )
    for finding in coverage.get("findings", []):
        tone = "danger" if finding["required_action"] == "emergency_escalation" else "warning"
        if finding["required_action"] == "adjuster_review":
            tone = "success"
        events.append(
            {
                "tone": tone,
                "title": finding["message"],
                "detail": f"Required action: {finding['required_action']}.",
                "rule": finding["rule_id"],
            }
        )
    for signal in fraud_gate.get("signals", []):
        tone = "danger" if signal.get("route_to_emergency") else "warning"
        events.append(
            {
                "tone": tone,
                "title": signal["message"],
                "detail": "Deterministic fraud/safety gate signal.",
                "rule": signal["signal_id"],
            }
        )
    route = fraud_gate.get("final_routing_decision", coverage.get("routing_decision"))
    if route != session.route:
        events.append(
            {
                "tone": "danger" if route == "emergency_escalation" else "success",
                "title": "Routing changed",
                "detail": f"{session.route} -> {route}.",
                "rule": "ROUTE-001",
            }
        )
    return events


def _next_question(
    validation: dict[str, Any],
    coverage: dict[str, Any],
    fraud_gate: dict[str, Any],
) -> str:
    route = fraud_gate.get("final_routing_decision", coverage.get("routing_decision"))
    signals = fraud_gate.get("signals", [])
    if route == "emergency_escalation" and any(
        signal.get("route_to_emergency") for signal in signals
    ):
        return (
            "Because you mentioned an injury or safety concern, are you and everyone "
            "else currently safe, and has anyone needed emergency medical care?"
        )

    missing = validation.get("missing_fields", [])
    question_map = {
        "policyholder_name": "What is your full name as it appears on the policy?",
        "contact_method": "What is the best phone number or email for the adjuster to reach you?",
        "date_of_loss": "When did the loss happen?",
        "loss_location": "Where did the loss happen?",
        "loss_description": "Can you briefly describe what happened?",
    }
    for field_name in missing:
        if field_name in question_map:
            return question_map[field_name]

    required_docs = coverage.get("required_documents", [])
    if required_docs:
        return f"Do you already have this document or evidence available: {required_docs[0]}?"

    return "I have enough for the initial intake packet. Is there anything important the adjuster should know before handoff?"


def _agent_reply(
    session: IntakeSession,
    validation: dict[str, Any],
    coverage: dict[str, Any],
    checklist: dict[str, Any],
    fraud_gate: dict[str, Any],
    packet: dict[str, Any],
) -> str:
    next_action = _next_question(validation, coverage, fraud_gate)
    route = fraud_gate.get("final_routing_decision", coverage.get("routing_decision"))
    prompt = f"""
You are the voice/text insurance FNOL intake agent speaking directly to the claimant.

Generate the next response for the claimant based on the latest turn, structured claim state,
deterministic rules, and next best action. Keep it concise, empathetic, and operational.

Hard constraints:
- Do not promise coverage, payment, liability, benefits, or claim approval.
- If injury, unsafe living condition, or immediate danger is present, prioritize safety and human review.
- Ask only one or two focused follow-up questions.
- Acknowledge facts already captured without repeating the full packet.
- Do not reveal hidden reasoning. You may reference that the intake packet was updated.

Transcript:
{session.transcript}

Normalized claim:
{session.normalized_claim}

Validation:
{validation}

Coverage/evidence rules:
{coverage}

Checklist:
{checklist}

Fraud/safety gate:
{fraud_gate}

Current routing decision:
{route}

Next best action:
{next_action}

Current handoff packet summary:
{packet.get("adjuster_handoff_summary")}
"""
    reply = _generate_structured(prompt, AgentReply)
    return reply["response_text"]


def _ui_state(
    session: IntakeSession,
    validation: dict[str, Any],
    coverage: dict[str, Any],
    checklist: dict[str, Any],
    fraud_gate: dict[str, Any],
    packet: dict[str, Any],
    events: list[dict[str, str]],
) -> dict[str, Any]:
    claim = ClaimNarrative.model_validate(_claim_from_session(session))
    classification = ClaimClassification.model_validate(session.classification)
    route = fraud_gate["final_routing_decision"]
    completed = 0

    def counted(field: dict[str, str]) -> dict[str, str]:
        nonlocal completed
        if field["status"] in {"complete", "urgent"}:
            completed += 1
        return field

    positive_safety_items = _positive_safety_items(claim.injuries_or_safety_concerns)
    safety_text = " ".join(
        [
            claim.loss_description,
            claim.raw_narrative_summary,
            _claimant_text(session),
            *claim.injuries_or_safety_concerns,
        ]
    )
    injury_text = _join(claim.injuries_or_safety_concerns, "Unknown")
    if not positive_safety_items and _has_negated_safety_mention(safety_text):
        injury_text = "No injuries reported"
    evidence_text = _join(claim.evidence_available, "Not captured yet")
    docs_text = _join(claim.documents_mentioned, "Not captured yet")
    required_doc_names = [item["item"] for item in checklist.get("items", [])]

    fields = {
        "claimant": counted(_field("Claimant name", claim.policyholder_name)),
        "policy": counted(_field("Policy number", claim.policy_number)),
        "contact": counted(_field("Contact method", claim.contact_method)),
        "type": counted(_field("Claim type", classification.claim_type.replace("_", " "))),
        "date": counted(_field("Date of loss", claim.date_of_loss)),
        "time": counted(_field("Reported date", claim.reported_date)),
        "location": counted(_field("Location", claim.loss_location)),
        "description": counted(_field("Loss description", claim.loss_description)),
        "injuries": counted(
            _field(
                "Injuries",
                injury_text,
                source="Gemini extraction + safety gate",
                urgent=bool(positive_safety_items),
            )
        ),
        "hazards": counted(_field("Hazards present", _join([item for item in claim.injuries_or_safety_concerns if "hazard" in item.lower() or "unsafe" in item.lower()], "Unknown"))),
        "medical": counted(_field("Medical attention", _join([item for item in claim.injuries_or_safety_concerns if "medical" in item.lower() or "care" in item.lower() or "hospital" in item.lower()], "Unknown"))),
        "police": counted(_field("Report number", _find_report(claim))),
        "photos": counted(_field("Evidence available", evidence_text)),
        "tow": counted(_field("Tow info", _find_text(claim, ["tow", "storage"]))),
        "otherDriver": counted(_field("Other driver info", _find_text(claim, ["other driver", "driver", "plate", "witness"]))),
    }

    progress = max(12, round(completed / len(fields) * 100))
    return {
        "route": route,
        "progress": progress,
        "fields": fields,
        "transcript": session.transcript,
        "events": events,
        "handoff": {
            "Summary": packet["adjuster_handoff_summary"],
            "Priority": f"{classification.severity.title()} - {classification.severity_rationale}",
            "Required actions": _join(required_doc_names, "No additional documents identified by current rules."),
            "Attachments": evidence_text,
            "Next best action": _next_question(validation, coverage, fraud_gate),
        },
        "packet_markdown": packet["markdown"],
        "model": MODEL,
    }


def _find_text(claim: ClaimNarrative, needles: list[str]) -> str:
    text = " | ".join(
        [claim.loss_description, *claim.evidence_available, *claim.documents_mentioned, *claim.parties_involved]
    )
    lower = text.lower()
    if any(needle in lower for needle in needles):
        return text
    return "not specified"


def _find_report(claim: ClaimNarrative) -> str:
    text = " | ".join([*claim.evidence_available, *claim.documents_mentioned, claim.loss_description])
    lower = text.lower()
    if any(term in lower for term in ["police", "report", "case number", "incident"]):
        return text
    return "not specified"


def _process(session: IntakeSession) -> dict[str, Any]:
    session.normalized_claim = _normalize_claim(session)
    validation = validate_required_claim_fields(session.normalized_claim)
    session.classification = _classify_claim(session.normalized_claim, validation)
    coverage = apply_coverage_and_evidence_rules(
        session.normalized_claim,
        validation,
        session.classification,
    )
    checklist = generate_document_checklist(
        session.normalized_claim,
        session.classification,
        coverage,
    )
    fraud_gate = fraud_signal_and_safety_gate(
        session.normalized_claim,
        validation,
        session.classification,
        coverage,
    )
    events = _events(session, validation, coverage, fraud_gate)
    session.route = fraud_gate["final_routing_decision"]
    packet = build_claim_intake_packet(
        session.normalized_claim,
        validation,
        session.classification,
        coverage,
        checklist,
        fraud_gate,
    )
    reply = _agent_reply(session, validation, coverage, checklist, fraud_gate, packet)
    session.transcript.append({"speaker": "Agent", "text": reply})
    return _ui_state(session, validation, coverage, checklist, fraud_gate, packet, events)


async def _process_live_state(session: IntakeSession) -> dict[str, Any]:
    session.normalized_claim = await _generate_structured_async(
        f"""
You are an insurance FNOL intake extraction service.

Use the full claimant transcript to produce one complete ClaimNarrative.
Preserve known facts from the current claim state unless the transcript corrects them.
Do not invent policy numbers, contacts, dates, locations, evidence, or dollar amounts.
Use "not specified" for unknown string fields.

Current claim state:
{_claim_from_session(session)}

Claimant transcript:
{_claimant_text(session)}
""",
        ClaimNarrative,
    )
    validation = validate_required_claim_fields(session.normalized_claim)
    session.classification = await _generate_structured_async(
        f"""
Classify this insurance FNOL intake for operational routing.
Return a ClaimClassification only. Do not decide coverage or payment.

Normalized claim:
{session.normalized_claim}

Field validation:
{validation}
""",
        ClaimClassification,
    )
    coverage = apply_coverage_and_evidence_rules(
        session.normalized_claim,
        validation,
        session.classification,
    )
    checklist = generate_document_checklist(
        session.normalized_claim,
        session.classification,
        coverage,
    )
    fraud_gate = fraud_signal_and_safety_gate(
        session.normalized_claim,
        validation,
        session.classification,
        coverage,
    )
    events = _events(session, validation, coverage, fraud_gate)
    session.route = fraud_gate["final_routing_decision"]
    packet = build_claim_intake_packet(
        session.normalized_claim,
        validation,
        session.classification,
        coverage,
        checklist,
        fraud_gate,
    )
    return _ui_state(session, validation, coverage, checklist, fraud_gate, packet, events)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"ok": True, "model": MODEL, "has_api_key": _has_api_key()}


@app.post("/api/sessions", response_model=SessionResponse)
def create_session() -> SessionResponse:
    session = IntakeSession(session_id=str(uuid.uuid4()))
    session.transcript.append(
        {
            "speaker": "Agent",
            "text": "I can start the claim while we talk. First, are you and everyone else in a safe place?",
        }
    )
    sessions[session.session_id] = session
    validation = validate_required_claim_fields(_blank_claim())
    classification = {
        "claim_type": "other",
        "severity": "medium",
        "severity_rationale": "Waiting for claimant facts.",
        "likely_policy_line": "unknown",
        "loss_drivers": [],
        "claimant_needs": ["Provide initial loss facts."],
    }
    session.normalized_claim = _blank_claim()
    session.classification = classification
    coverage = apply_coverage_and_evidence_rules(session.normalized_claim, validation, classification)
    checklist = generate_document_checklist(session.normalized_claim, classification, coverage)
    fraud_gate = fraud_signal_and_safety_gate(
        session.normalized_claim, validation, classification, coverage
    )
    packet = build_claim_intake_packet(
        session.normalized_claim,
        validation,
        classification,
        coverage,
        checklist,
        fraud_gate,
    )
    state = _ui_state(
        session,
        validation,
        coverage,
        checklist,
        fraud_gate,
        packet,
        [
            {
                "tone": "warning",
                "title": "Waiting for claimant facts",
                "detail": "The backend session is open and ready for Gemini extraction.",
                "rule": "SESSION-001",
            }
        ],
    )
    return SessionResponse(
        session_id=session.session_id,
        model=MODEL,
        has_api_key=_has_api_key(),
        state=state,
    )


@app.post("/api/message", response_model=SessionResponse)
def message(request: MessageRequest) -> SessionResponse:
    session = sessions.get(request.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown intake session.")
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Message text is required.")
    session.transcript.append({"speaker": "Claimant", "text": text})
    state = _process(session)
    return SessionResponse(
        session_id=session.session_id,
        model=MODEL,
        has_api_key=_has_api_key(),
        state=state,
    )


@app.post("/api/audio", response_model=SessionResponse)
async def audio_message(
    session_id: str = Form(...),
    audio: UploadFile = File(...),
) -> SessionResponse:
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown intake session.")
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Audio file is empty.")
    mime_type = audio.content_type or "audio/webm"
    text = _transcribe_audio(audio_bytes, mime_type)
    session.transcript.append({"speaker": "Claimant", "text": text})
    state = _process(session)
    return SessionResponse(
        session_id=session.session_id,
        model=MODEL,
        has_api_key=_has_api_key(),
        state=state,
    )


@app.websocket("/ws/live")
async def live_voice(websocket: WebSocket) -> None:
    await websocket.accept()
    session_id = str(uuid.uuid4())
    session = IntakeSession(session_id=session_id)
    session.transcript.append(
        {
            "speaker": "Agent",
            "text": "I can start the claim while we talk. First, are you and everyone else in a safe place?",
        }
    )
    sessions[session_id] = session

    try:
        from google.genai import types
    except ImportError:
        await websocket.send_json(
            {"type": "error", "message": "Missing google-genai package. Run pip install -r requirements.txt."}
        )
        await websocket.close()
        return

    if not _has_api_key():
        await websocket.send_json(
            {
                "type": "error",
                "message": f"Missing GOOGLE_API_KEY. Add it to {APP_DIR / '.env'} and restart the backend.",
            }
        )
        await websocket.close()
        return

    await websocket.send_json(
        {
            "type": "session",
            "session_id": session_id,
            "model": LIVE_MODEL,
            "message": "Gemini Live voice session connected.",
        }
    )

    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=(
            "You are a voice insurance FNOL intake agent. Speak naturally and briefly. "
            "Collect claim facts one step at a time. If injury, unsafe housing, or immediate "
            "danger is mentioned, prioritize safety and human escalation. Do not promise "
            "coverage, payment, liability, benefits, or approval. Ask only one or two focused "
            "follow-up questions at a time."
        ),
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Kore")
            )
        ),
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
    )

    state_lock = asyncio.Lock()

    async def update_claim_state(text: str) -> None:
        if not text.strip():
            return
        async with state_lock:
            session.transcript.append({"speaker": "Claimant", "text": text.strip()})
            try:
                state = await _process_live_state(session)
                await websocket.send_json({"type": "state", "state": state})
            except Exception as exc:
                await websocket.send_json({"type": "error", "message": f"Claim state update failed: {exc}"})

    try:
        async with _client().aio.live.connect(model=LIVE_MODEL, config=config) as live_session:
            async def client_to_gemini() -> None:
                while True:
                    message = await websocket.receive_json()
                    msg_type = message.get("type")
                    if msg_type == "audio":
                        data = base64.b64decode(message["data"])
                        await live_session.send_realtime_input(
                            audio=types.Blob(data=data, mime_type="audio/pcm;rate=16000")
                        )
                    elif msg_type == "text":
                        text = str(message.get("text", "")).strip()
                        if text:
                            session.transcript.append({"speaker": "Claimant", "text": text})
                            await live_session.send(input=text, end_of_turn=True)
                            state = await _process_live_state(session)
                            await websocket.send_json({"type": "state", "state": state})
                    elif msg_type == "close":
                        await websocket.close()
                        return

            async def gemini_to_client() -> None:
                pending_input = ""
                pending_output = ""

                async def finalize_input(reason: str) -> None:
                    nonlocal pending_input
                    finished = pending_input.strip()
                    if not finished:
                        return
                    pending_input = ""
                    await websocket.send_json(
                        {
                            "type": "transcript",
                            "speaker": "Claimant",
                            "text": finished,
                            "final": True,
                            "reason": reason,
                        }
                    )
                    asyncio.create_task(update_claim_state(finished))

                async def finalize_output(reason: str) -> None:
                    nonlocal pending_output
                    finished = pending_output.strip()
                    if not finished:
                        return
                    pending_output = ""
                    session.transcript.append({"speaker": "Agent", "text": finished})
                    await websocket.send_json(
                        {
                            "type": "transcript",
                            "speaker": "Agent",
                            "text": finished,
                            "final": True,
                            "reason": reason,
                        }
                    )

                while True:
                    turn = live_session.receive()
                    async for response in turn:
                        server_content = response.server_content
                        if not server_content:
                            continue

                        if server_content.input_transcription and server_content.input_transcription.text:
                            text = server_content.input_transcription.text
                            pending_input += text
                            await websocket.send_json(
                                {
                                    "type": "transcript",
                                    "speaker": "Claimant",
                                    "text": pending_input,
                                    "delta": text,
                                    "final": bool(getattr(server_content.input_transcription, "finished", False)),
                                }
                            )
                            if getattr(server_content.input_transcription, "finished", False):
                                await finalize_input("input_transcription_finished")

                        if server_content.output_transcription and server_content.output_transcription.text:
                            await finalize_input("model_started_response")
                            text = server_content.output_transcription.text
                            pending_output += text
                            await websocket.send_json(
                                {
                                    "type": "transcript",
                                    "speaker": "Agent",
                                    "text": pending_output,
                                    "delta": text,
                                    "final": bool(getattr(server_content.output_transcription, "finished", False)),
                                }
                            )
                            if getattr(server_content.output_transcription, "finished", False):
                                await finalize_output("output_transcription_finished")

                        if server_content.model_turn:
                            await finalize_input("model_audio_started")
                            for part in server_content.model_turn.parts or []:
                                if part.inline_data and isinstance(part.inline_data.data, bytes):
                                    await websocket.send_json(
                                        {
                                            "type": "audio",
                                            "data": base64.b64encode(part.inline_data.data).decode("ascii"),
                                            "mime_type": part.inline_data.mime_type or "audio/pcm;rate=24000",
                                        }
                                    )

                        if server_content.interrupted:
                            pending_output = ""
                            await websocket.send_json({"type": "interrupted"})

                        if (
                            getattr(server_content, "generation_complete", False)
                            or getattr(server_content, "turn_complete", False)
                            or getattr(server_content, "waiting_for_input", False)
                        ):
                            await finalize_output("live_turn_complete")

            await asyncio.gather(client_to_gemini(), gemini_to_client())
    except WebSocketDisconnect:
        return
    except Exception as exc:
        print(f"Gemini Live session failed: {type(exc).__name__}: {exc}", flush=True)
        try:
            await websocket.send_json({"type": "error", "message": f"Gemini Live session failed: {exc}"})
        except Exception:
            pass


@app.get("/")
def index() -> FileResponse:
    return FileResponse(DEMO_DIR / "index.html")


app.mount("/", StaticFiles(directory=DEMO_DIR, html=True), name="static")
