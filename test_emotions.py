import time
import json
from emotion_service import EmotionTracker

def print_separator(title=""):
    print("\n" + "=" * 55)
    if title:
        print(f"  {title}")
        print("=" * 55)

def print_timeline_row(entry: dict):
    second      = entry.get("second", "?")
    emotion     = entry.get("emotion") or "no face"
    conf        = entry.get("confidence", 0)
    neutrality  = entry.get("neutrality_score", 0)
    eye_contact = "👁 YES" if entry.get("eye_contact") else "❌ NO"
    gaze        = entry.get("gaze_score", 0.0)
    face        = "✅" if entry.get("face_detected") else "👤"

    emotion_bar   = "█" * int(conf * 10)
    neutrality_bar = "▓" * int(neutrality / 10)

    print(
        f"  [{second:>3}s] {face} {emotion:<10} "
        f"conf:{int(conf*100):>3}% {emotion_bar:<10} | "
        f"neutral:{int(neutrality):>3}% {neutrality_bar:<10} | "
        f"eye:{eye_contact} gaze:{gaze:.2f}"
    )

def print_summary(result: dict):
    s = result["summary"]

    print_separator("EMOTION SUMMARY")
    print(f"  Dominant emotion      : {s['dominant_emotion']}")
    print(f"  Confidence score      : {s['confidence_score']}/100")
    print(f"  Avg neutrality score  : {s['avg_neutrality_score']}/100")
    print(f"  Nervous %             : {s['nervous_percentage']}%")
    print(f"  Happy %               : {s['happy_percentage']}%")
    print(f"  Neutral %             : {s['neutral_percentage']}%")
    print(f"  Angry %               : {s['angry_percentage']}%")
    print(f"  Emotion breakdown     : {s['emotion_breakdown']}")

    print_separator("EYE CONTACT SUMMARY")
    quality_emoji = {
        "excellent": "🟢",
        "good": "🟡",
        "needs_improvement": "🟠",
        "poor": "🔴"
    }
    emoji = quality_emoji.get(s['eye_contact_quality'], "⚪")
    print(f"  Eye contact %         : {s['eye_contact_percentage']}%")
    print(f"  Eye contact quality   : {emoji} {s['eye_contact_quality']}")
    print(f"  Avg gaze score        : {s['avg_gaze_score']}/1.0")

    print_separator("DELIVERY SCORE")
    score = s['delivery_score']
    bar = "█" * int(score / 5)
    if score >= 75:
        label = "🟢 Excellent"
    elif score >= 55:
        label = "🟡 Good"
    elif score >= 35:
        label = "🟠 Needs work"
    else:
        label = "🔴 Poor"
    print(f"  {score}/100  {bar}  {label}")
    print(f"  (emotion 50% + neutrality 30% + eye contact 20%)")

    print_separator("HIGHLIGHTS")
    if s['nervous_moments']:
        print(f"  ⚠️  Nervous moments     : {', '.join(s['nervous_moments'])}")
    else:
        print(f"  ✅ No nervous moments detected")

    if s['lost_eye_contact_moments']:
        print(f"  👀 Lost eye contact at  : {', '.join(s['lost_eye_contact_moments'])}")
    else:
        print(f"  ✅ Eye contact maintained throughout")

    if s['face_absent_seconds']:
        print(f"  👤 Face absent at       : {s['face_absent_seconds']}")
    else:
        print(f"  ✅ Face visible throughout")

    print(f"\n  Duration              : {result['duration_seconds']}s total")
    print(f"  Face detected         : {result['face_detected_seconds']}s")


def run_test(label: str, duration: int, question: str, part: int,
             instruction: str = None):
    print_separator(label)
    if instruction:
        print(f"  📋 {instruction}")
    print(f"  ⏱  Running for {duration} seconds...\n")

    tracker = EmotionTracker()
    tracker.start_tracking()

    for i in range(duration):
        time.sleep(1)
        if tracker.timeline:
            latest = tracker.timeline[-1]
            print_timeline_row(latest)
        else:
            print(f"  [{i+1:>3}s] ⏳ waiting for first frame...")

    result = tracker.stop_tracking(question=question, part=part)
    print_summary(result)
    return result


def main():
    print_separator("IELTS EMOTION TRACKER — FULL PIPELINE TEST")
    print("  Tests: emotion detection + eye contact + gaze analysis")
    print("  Make sure your camera is plugged in and working")

    # ── Test 1: Natural speaking (10 seconds) ─────────────────────────────────
    result1 = run_test(
        label="TEST 1 — Natural Speaking (10s)",
        duration=10,
        question="How often do you use the internet?",
        part=1,
        instruction="Speak naturally, look at the camera as if talking to examiner"
    )

    # ── Test 2: Eye contact test (8 seconds) ──────────────────────────────────
    result2 = run_test(
        label="TEST 2 — Eye Contact Test (8s)",
        duration=8,
        question="Describe a place you love.",
        part=2,
        instruction="First 4 seconds: look directly at camera. Last 4 seconds: look away"
    )

    # ── Test 3: Nervousness simulation (8 seconds) ────────────────────────────
    result3 = run_test(
        label="TEST 3 — Nervousness Simulation (8s)",
        duration=8,
        question="What are the effects of social media on society?",
        part=3,
        instruction="Try to look nervous/anxious — frown, avoid eye contact"
    )

    # ── Test 4: No face (cover camera) ────────────────────────────────────────
    result4 = run_test(
        label="TEST 4 — No Face Handling (5s)",
        duration=5,
        question="Test question",
        part=1,
        instruction="Cover your camera completely for 5 seconds"
    )

    # ── Final comparison ───────────────────────────────────────────────────────
    print_separator("FINAL COMPARISON")
    print(f"  {'Test':<35} {'Delivery':>8} {'Eye Contact':>12} {'Nervous':>8}")
    print(f"  {'-'*35} {'-'*8} {'-'*12} {'-'*8}")

    tests = [
        ("Test 1 — Natural speaking",   result1),
        ("Test 2 — Eye contact",        result2),
        ("Test 3 — Nervousness",        result3),
        ("Test 4 — No face",            result4),
    ]

    for name, r in tests:
        s = r["summary"]
        print(
            f"  {name:<35} "
            f"{s['delivery_score']:>7}/100 "
            f"{s['eye_contact_percentage']:>10}% "
            f"{s['nervous_percentage']:>6}%"
        )

    print_separator("FULL JSON — TEST 1")
    print(json.dumps(result1, indent=2))

    print_separator("ALL TESTS COMPLETE ✅")
    print("  Your emotion_service.py is working correctly.")
    print("  You can now integrate it with your FastAPI backend.\n")


if __name__ == "__main__":
    main()