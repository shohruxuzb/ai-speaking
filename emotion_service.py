import cv2
import time
import threading
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()

EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

class EmotionTracker:
    def __init__(self):
        self.recognizer = HSEmotionRecognizer(model_name='enet_b0_8_best_afew')
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.timeline = []
        self.is_running = False
        self.second = 0
        self._thread = None
        self.cap = None

    def start_tracking(self):
        self.cap = cv2.VideoCapture(0)
        self.timeline = []
        self.second = 0
        self.is_running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def _sample_loop(self):
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(1)
                continue
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    face_img = frame[y:y+h, x:x+w]
                    emotion, scores = self.recognizer.predict_emotions(face_img, logits=False)
                    dominant = emotion.lower()
                    emotions = {label: round(float(score), 2) for label, score in zip(EMOTION_LABELS, scores)}
                    self.timeline.append({
                        "second": self.second,
                        "emotion": dominant,
                        "confidence": round(emotions[dominant], 2),
                        "all": emotions
                    })
            except Exception:
                pass

            self.second += 1
            time.sleep(1)

    def stop_tracking(self, question: str = "", part: int = 1) -> dict:
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=3)
        if self.cap:
            self.cap.release()
        return self._build_result(question, part)

    def _build_result(self, question: str, part: int) -> dict:
        timeline = self.timeline
        total = len(timeline) or 1

        emotion_counts = {}
        for entry in timeline:
            em = entry['emotion']
            emotion_counts[em] = emotion_counts.get(em, 0) + 1

        dominant = max(emotion_counts, key=emotion_counts.get) if emotion_counts else 'neutral'
        nervous_pct = ((emotion_counts.get('fear', 0) + emotion_counts.get('sad', 0)) / total) * 100
        happy_pct = (emotion_counts.get('happy', 0) / total) * 100
        neutral_pct = (emotion_counts.get('neutral', 0) / total) * 100
        conf_score = min(100, max(0, round(happy_pct * 0.8 + neutral_pct * 0.5 - nervous_pct * 0.5 + 50)))

        return {
            "part": part,
            "question": question,
            "duration_seconds": total,
            "emotion_timeline": timeline,
            "summary": {
                "dominant_emotion": dominant,
                "confidence_score": conf_score,
                "nervous_percentage": round(nervous_pct),
                "happy_percentage": round(happy_pct),
                "emotion_breakdown": emotion_counts
            }
        }


tracker = EmotionTracker()

@router.post("/emotion/start")
async def start_emotion_tracking():
    tracker.start_tracking()
    return {"status": "tracking started"}

@router.post("/emotion/stop")
async def stop_emotion_tracking(question: str = "", part: int = 1):
    result = tracker.stop_tracking(question=question, part=part)
    return JSONResponse(content=result)