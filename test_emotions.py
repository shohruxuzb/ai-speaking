import time
import json
from emotion_service import EmotionTracker


def print_separator(title=""):
    print("\n" + "=" * 50)
    if title:
        print(f"  {title}")
        print("=" * 50)


def test_emotion_tracker():
    print_separator("EMOTION TRACKER TEST PIPELINE")
    print("Initializing FER detector... (may take a few seconds)")

    tracker = EmotionTracker()
    print("✅ Tracker initialized")

    # ── Test 1: Short 5-second tracking ──────────────────
    print_separator("TEST 1: 5-second tracking")
    print("📷 Starting camera... look at the camera!")
    print("Tracking for 5 seconds...\n")

    tracker.start_tracking()

    # Live feedback while tracking
    for i in range(5):
        time.sleep(1)
        samples = len(tracker.timeline)
        if samples > 0:
            latest = tracker.timeline[-1]
            print(f"  ⏱  Second {i + 1}: emotion={latest['emotion']} | confidence={latest['confidence']}")
        else:
            print(f"  ⏱  Second {i + 1}: no face detected")

    result = tracker.stop_tracking(
        question="How often do you use the internet?",
        part=1
    )
    print("\n✅ Tracking stopped")

    # ── Print Results ─────────────────────────────────────
    print_separator("RESULTS")

    print(f"  Question   : {result['question']}")
    print(f"  Part       : {result['part']}")
    print(f"  Duration   : {result['duration_seconds']} seconds")

    print_separator("SUMMARY")
    s = result['summary']
    print(f"  Dominant emotion  : {s['dominant_emotion']}")
    print(f"  Confidence score  : {s['confidence_score']}/100")
    print(f"  Nervous %         : {s['nervous_percentage']}%")
    print(f"  Happy %           : {s['happy_percentage']}%")
    print(f"  Emotion breakdown : {s['emotion_breakdown']}")

    print_separator("EMOTION TIMELINE")
    if result['emotion_timeline']:
        for entry in result['emotion_timeline']:
            bar = "█" * int(entry['confidence'] * 20)
            print(f"  [{entry['second']:>2}s] {entry['emotion']:<10} {bar} {int(entry['confidence'] * 100)}%")
    else:
        print("  ⚠️  No emotions detected — check camera or lighting")

    print_separator("FULL JSON OUTPUT")
    print(json.dumps(result, indent=2))

    # ── Test 2: No face test ──────────────────────────────
    print_separator("TEST 2: No face detection (cover camera)")
    print("Cover your camera for 3 seconds...")

    tracker2 = EmotionTracker()
    tracker2.start_tracking()
    time.sleep(3)
    result2 = tracker2.stop_tracking(question="Test question", part=2)

    print(f"  Samples collected: {len(result2['emotion_timeline'])}")
    print(f"  Dominant: {result2['summary']['dominant_emotion']}")
    print("✅ Gracefully handled no-face scenario")

    # ── Test 3: Simulate different emotions ──────────────
    print_separator("TEST 3: 10-second full test")
    print("Look at camera naturally for 10 seconds — try smiling, then neutral\n")

    tracker3 = EmotionTracker()
    tracker3.start_tracking()

    for i in range(10):
        time.sleep(1)
        samples = len(tracker3.timeline)
        if samples > 0:
            latest = tracker3.timeline[-1]
            all_em = latest.get('all', {})
            top3 = sorted(all_em.items(), key=lambda x: x[1], reverse=True)[:3]
            top3_str = " | ".join([f"{k}:{int(v * 100)}%" for k, v in top3])
            print(f"  ⏱  {i + 1:>2}s | {latest['emotion']:<10} | {top3_str}")
        else:
            print(f"  ⏱  {i + 1:>2}s | no face detected")

    result3 = tracker3.stop_tracking(question="Describe a place you love", part=2)

    print(f"\n  Final confidence score: {result3['summary']['confidence_score']}/100")
    print(f"  Dominant emotion: {result3['summary']['dominant_emotion']}")

    print_separator("ALL TESTS COMPLETE ✅")


if __name__ == "__main__":
    test_emotion_tracker()