from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import tempfile
import uvicorn
import json
import os
import io
import random
import re

from dotenv import load_dotenv
from groq import Groq
from typing import List, Dict
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings

load_dotenv()

# Init app
app = FastAPI()

try:
    from emotion_service import router as emotion_router
    app.include_router(emotion_router)
except ImportError:
    print("Warning: emotion_service.py not found or has errors.")


@app.get("/")
async def root():
    return {"status": "ok"}


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
elevenlabs_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

# Examiner voice — change voice_id to switch voice:
# Rachel (neutral):  21m00Tcm4TlvDq8ikWAM
# Dorothy (British): ThT5KcBeYPX3keUQqHPh
# Alice (warm GB):   Xb7hH8MSUJpSbSDYk0k2
# Daniel (British M):onwK4e9ZLuTAKqWW03F9
EXAMINER_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"


def transcribe_audio(audio_path: str) -> str:
    """Transcribe speech using Whisper on Groq — FIX: language hint + deterministic"""
    with open(audio_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=f,
            language="en",                        # force English recognition
            prompt="IELTS speaking test answer.",  # context hint improves accuracy
            temperature=0.0,                       # deterministic output
        )
    return transcript.text if transcript else ""


def _delivery_score_to_band(score: float) -> str:
    """Convert 0-100 delivery score to IELTS-style band label."""
    if score >= 85: return "Excellent Delivery"
    if score >= 70: return "Good Delivery"
    if score >= 55: return "Satisfactory Delivery"
    if score >= 40: return "Needs Improvement"
    return "Poor Delivery"


def evaluate_ielts_with_improvements(
        questions: List[str],
        answers: List[str],
        emotion_data: dict = None
) -> dict:
    qa_pairs = "\n".join(
        [f"{i + 1}. Q: \"{q}\" \n   A: \"{a}\"" for i, (q, a) in enumerate(zip(questions, answers))]
    )

    # ── Step 1: Calculate emotion adjustment BEFORE calling LLM ───────────────
    emotion_adjustment = 0.0
    delivery_score     = None
    emotion_context    = ""

    if emotion_data and "summary" in emotion_data:
        s = emotion_data["summary"]

        nervous_pct     = s.get("nervous_percentage", 0)
        happy_pct       = s.get("happy_percentage", 0)
        neutral_pct     = s.get("neutral_percentage", 0)
        eye_contact_pct = s.get("eye_contact_percentage", 0)
        delivery_score  = s.get("delivery_score", None)

        # Fluency adjustment based on nervousness
        if nervous_pct > 70:
            emotion_adjustment = -1.0
        elif nervous_pct > 50:
            emotion_adjustment = -0.5
        elif nervous_pct > 30:
            emotion_adjustment = -0.25
        elif happy_pct + neutral_pct > 60:
            emotion_adjustment = +0.5

        # Poor eye contact penalises coherence
        if eye_contact_pct < 25:
            emotion_adjustment -= 0.25

        timeline = emotion_data.get("emotion_timeline", [])
        nervous_moments = [
            f"second {e['second']}"
            for e in timeline
            if e.get("emotion") in ("fear", "sad") and e.get("confidence", 0) > 0.5
        ]
        nervous_str = ", ".join(nervous_moments[:5]) or "none"

        emotion_context = f"""
EMOTION ANALYSIS during speaking:
- Dominant emotion:  {s.get("dominant_emotion")}
- Nervous/anxious:   {nervous_pct}% of time
- Happy/positive:    {happy_pct}% of time
- Eye contact:       {eye_contact_pct}% of time ({s.get("eye_contact_quality")})
- Delivery score:    {delivery_score}/100
- Peak nervousness at: {nervous_str}

FLUENCY ADJUSTMENT from emotion: {emotion_adjustment:+.2f} bands
Apply this adjustment to your fluency score.
- Nervous > 50%: candidate likely hesitated more than text shows — reduce fluency
- Eye contact poor: shows lack of engagement — reduce coherence
- Confident/happy: candidate likely sounded more fluent — boost fluency
"""

    # ── Step 2: Answer length quality check ───────────────────────────────────
    quality_notes = []
    for i, answer in enumerate(answers):
        word_count = len(answer.split())
        if word_count < 20:
            quality_notes.append(f"Answer {i+1}: very short ({word_count} words) — likely band 4-5")
        elif word_count < 40:
            quality_notes.append(f"Answer {i+1}: short ({word_count} words) — cap at band 6")
    quality_str = "\n".join(quality_notes) if quality_notes else "All answers are adequate length."

    prompt = f"""
You are a strict certified IELTS examiner. Be realistic and critical.

BAND SCORE GUIDELINES:
- Band 4: Many errors, very limited vocabulary, frequent long pauses
- Band 5: Noticeable errors, limited vocabulary, some hesitation
- Band 6: Some errors, adequate vocabulary, occasional hesitation
- Band 7: Few errors, good vocabulary, mostly fluent
- Band 8: Rare errors, wide vocabulary, very fluent
- Band 9: No errors, extensive vocabulary, completely fluent

STRICT RULES:
- Most candidates score 5.0-7.0 — do NOT inflate scores
- Deduct 0.5 for: filler words, very short answers, repeated vocabulary
- Only give 7.5+ if genuinely impressive vocabulary and complex grammar
- Pronunciation defaults to 6.0 (cannot be scored from text alone)
{f"- Apply the emotion-based fluency adjustment shown below" if emotion_context else ""}

Answer length assessment:
{quality_str}
{emotion_context}

Candidate responses:
{qa_pairs}

Return ONLY valid JSON:
{{
  "overall_band": float,
  "fluency": float,
  "vocabulary": float,
  "grammar": float,
  "pronunciation": float,
  "strengths": [list of strings],
  "weaknesses": [list of strings],
  "improved_answers": [list same length as input],
  "emotion_feedback": "specific one-sentence coaching tip based on emotion data"
}}
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=1000,
        )
        text_output = response.choices[0].message.content
        result_json = json.loads(text_output[text_output.index("{"): text_output.rindex("}") + 1])

        # ── Step 3: Apply emotion adjustment directly to fluency score ─────────
        if emotion_adjustment != 0.0:
            original_fluency = result_json.get("fluency", 6.0)
            adjusted = max(1.0, min(9.0, original_fluency + emotion_adjustment))
            # Round to nearest 0.5
            result_json["fluency"] = round(adjusted * 2) / 2

            # Recalculate overall band with adjusted fluency
            criteria = ["fluency", "vocabulary", "grammar", "pronunciation"]
            scores   = [result_json.get(c, 6.0) for c in criteria]
            result_json["overall_band"] = round(round(sum(scores) / len(scores) * 2) / 2, 1)

            result_json["fluency_emotion_adjustment"] = emotion_adjustment

        # ── Step 4: Add delivery score as unique 5th metric ───────────────────
        if delivery_score is not None:
            result_json["delivery_score"] = delivery_score
            result_json["delivery_band"]  = _delivery_score_to_band(delivery_score)

        return result_json

    except Exception as e:
        return {"error": str(e)}


@app.post("/evaluate")
async def evaluate(
    questions: str = Form(...),
    answers: str = Form(...),
    audios: List[UploadFile] = File(None)
):
    questions = json.loads(questions) if isinstance(questions, str) else questions
    answers = json.loads(answers) if isinstance(answers, str) else answers

    if audios:
        transcripts = []
        for audio in audios:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(await audio.read())
                tmp_path = tmp.name
            try:
                transcripts.append(transcribe_audio(tmp_path))
            finally:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
        answers = transcripts

    if not questions or not answers or len(questions) != len(answers):
        return {"error": "Questions and answers must be same length and non-empty"}

    evaluation = evaluate_ielts_with_improvements(questions, answers)
    return {"answers": answers, "evaluation": evaluation}


# ─────────────────────────────────────────────
# FIX 1: Separate used sets per part
# FIX 2: Track entire batches, not just single questions
# FIX 3: Reset threshold lowered so pool refreshes more naturally
# ─────────────────────────────────────────────
asked_questions = {1: set(), 2: set(), 3: set()}

# Large diverse topic lists per part to force variety
PART1_TOPICS = [
    "hometown", "hobbies", "food", "family", "friends", "daily routine",
    "weather", "sports", "music", "reading", "shopping", "cooking",
    "travel", "technology", "social media", "transport", "animals",
    "clothes", "morning routine", "weekends", "neighbors", "school",
    "work", "languages", "art", "films", "celebrations", "sleep",
    "walking", "nature", "gardens", "gifts", "photos", "memories",
]

PART2_TOPICS = [
    "a person who inspired you", "a memorable journey", "a skill you learned",
    "a book that influenced you", "a place you love", "a challenging experience",
    "a childhood memory", "a teacher who helped you", "a piece of technology",
    "a celebration you enjoyed", "a sport or game", "a piece of art",
    "a time you helped someone", "a goal you achieved", "a historical place",
    "a gift you received", "a festival in your country", "a change in your life",
]

PART3_TOPICS = [
    "technology and society", "education systems", "environmental issues",
    "globalization", "healthcare", "urbanization", "cultural preservation",
    "social media impact", "gender equality", "economic development",
    "immigration", "work-life balance", "mental health awareness",
    "role of government", "youth and society", "media influence",
    "traditional vs modern values", "climate change solutions",
]


def normalize_question(q: str) -> str:
    """Normalize a question to detect duplicates."""
    q = q.lower().strip()
    q = re.sub(
        r'^(can you|could you|do you think|describe|talk about|tell me about|would you rather|if you could)\s+',
        '', q,
    )
    q = re.sub(r'[?.!,]', '', q)
    return q.strip()


def _generate_unique_question(prompt: str, part: int) -> str:
    """Generate a unique question for given part, with retry logic."""
    global asked_questions

    # Reset if pool getting large
    if len(asked_questions[part]) > 150:
        asked_questions[part].clear()

    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=1.0,
                top_p=1,
                presence_penalty=1.5,   # FIX: increased to push for novelty
                frequency_penalty=1.0,  # FIX: increased to avoid repetition
                max_tokens=150,
            )
            question = response.choices[0].message.content.strip()
            # Remove quotes if LLM wrapped in them
            question = question.strip('"\'')
            normalized = normalize_question(question)

            if normalized not in asked_questions[part]:
                asked_questions[part].add(normalized)
                return question

        except Exception as e:
            print(f"Generation attempt {attempt+1} failed: {e}")

    # Last resort fallback
    return question


def generate_part1_questions() -> dict:
    """Generate 4 IELTS Part 1 questions on completely different topics."""

    # FIX: Pick 4 different random topics upfront, force LLM to use each
    topics = random.sample(PART1_TOPICS, 4)
    questions = []

    for topic in topics:
        prompt = f"""
Generate ONE IELTS Speaking Part 1 question about the topic: "{topic}"

Requirements:
- Short and direct, suitable for a 20-30 second answer
- Must be specifically about: {topic}
- Do NOT generate questions about these already-used topics this session: {list(asked_questions[1])}
- Return ONLY the question text, no explanations
- Do not include quotes around the question
"""
        q = _generate_unique_question(prompt, part=1)
        questions.append(q)

    return {"questions": questions}


def generate_part2_question() -> dict:
    """Generate 1 IELTS Part 2 cue card on a unique topic."""

    topic = random.choice([t for t in PART2_TOPICS
                           if normalize_question(t) not in asked_questions[2]]
                          or PART2_TOPICS)  # fallback to full list if all used

    prompt = f"""
Generate ONE IELTS Speaking Part 2 cue card question about: "{topic}"

Requirements:
- Start with "Describe ..." or "Talk about ..."
- Include exactly 3 bullet points using dashes (-)
- Suitable for a 1-2 minute answer
- Topic: {topic}
- Return ONLY the cue card text with bullet points
- Do not include quotes
"""
    question = _generate_unique_question(prompt, part=2)
    return {"question": question}


def generate_part3_questions() -> dict:
    """Generate 3 IELTS Part 3 questions on different abstract topics."""

    # FIX: Pick 3 different abstract topics upfront
    topics = random.sample(PART3_TOPICS, 3)
    questions = []

    for topic in topics:
        prompt = f"""
Generate ONE IELTS Speaking Part 3 discussion question about: "{topic}"

Requirements:
- Abstract and opinion-based, suitable for 30-40 second answer
- Must be specifically about: {topic}
- Do NOT repeat these already-asked topics: {list(asked_questions[3])}
- Return ONLY the question text
- Do not include quotes around the question
"""
        q = _generate_unique_question(prompt, part=3)
        questions.append(q)

    return {"questions": questions}


# ─── API ROUTES ───────────────────────────────

@app.get("/generate-part1")
async def generate_part1():
    return generate_part1_questions()

@app.get("/generate-part2")
async def generate_part2():
    return generate_part2_question()

@app.get("/generate-part3")
async def generate_part3():
    return generate_part3_questions()


# ─── AGGREGATE RESULTS ────────────────────────

def aggregate_evaluations(evaluations: list[dict]) -> dict:
    if not evaluations or len(evaluations) != 3:
        return {"error": "Expected exactly 3 evaluations (Part 1, Part 2, Part 3)"}

    weights = [0.25, 0.40, 0.35]
    criteria = ["fluency", "vocabulary", "grammar", "pronunciation", "overall_band"]

    weighted_scores = {c: 0.0 for c in criteria}
    strengths, weaknesses = [], []

    for i, ev in enumerate(evaluations):
        if "error" in ev:
            continue
        try:
            for c in criteria:
                weighted_scores[c] += float(ev[c]) * weights[i]
            strengths.extend(ev.get("strengths", []))
            weaknesses.extend(ev.get("weaknesses", []))
        except Exception:
            continue

    return {
        "overall_band": round(weighted_scores["overall_band"], 1),
        "fluency": round(weighted_scores["fluency"], 1),
        "vocabulary": round(weighted_scores["vocabulary"], 1),
        "grammar": round(weighted_scores["grammar"], 1),
        "pronunciation": round(weighted_scores["pronunciation"], 1),
        "strengths": list(set(strengths)),
        "weaknesses": list(set(weaknesses))
    }


class EvaluationsRequest(BaseModel):
    evaluations: List[Dict]

@app.post("/aggregate-results")
async def aggregate_results(req: EvaluationsRequest):
    return aggregate_evaluations(req.evaluations)


# ─── TTS / MOCK TEST ──────────────────────────

def synthesize_speech(text: str) -> bytes:
    """Convert text to speech using ElevenLabs."""
    try:
        audio_generator = elevenlabs_client.text_to_speech.convert(
            voice_id=EXAMINER_VOICE_ID,
            text=text,
            model_id="eleven_turbo_v2",
            voice_settings=VoiceSettings(
                stability=0.75,
                similarity_boost=0.85,
                style=0.40,
                use_speaker_boost=True
            )
        )
        return b"".join(chunk for chunk in audio_generator)
    except Exception as e:
        print(f"ElevenLabs TTS error: {e}")
        return b""


@app.post("/mock-test-voice")
async def mock_test_voice(part1: list[str], part2: str, part3: list[str]):
    script = []
    script.append("Hello, welcome to your IELTS speaking mock test. Let's begin.")
    script.append("Part one. I will ask you some questions about yourself and everyday topics.")
    for q in part1:
        script.append(q)
    script.append("Now let's move on to part two. You will have a cue card.")
    script.append(part2)
    script.append("Now we continue with part three. I will ask you some more discussion questions.")
    for q in part3:
        script.append(q)
    script.append("That concludes the IELTS speaking mock test. Thank you, and goodbye.")

    full_text = " ".join(script)
    audio_bytes = synthesize_speech(full_text)

    if not audio_bytes:
        return {"error": "TTS generation failed"}

    return StreamingResponse(io.BytesIO(audio_bytes), media_type="audio/mpeg")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)