import cv2
import time
import threading
import numpy as np
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
from fastapi import APIRouter
from fastapi.responses import JSONResponse

try:
    import mediapipe as mp
    _ = mp.solutions.face_mesh  # verify solutions API exists
    MEDIAPIPE_AVAILABLE = True
except (ImportError, AttributeError):
    MEDIAPIPE_AVAILABLE = False
    print("Warning: mediapipe not available — falling back to OpenCV gaze detection")
    print("Fix: pip uninstall mediapipe -y && pip install mediapipe==0.10.9")

router = APIRouter()

EMOTION_LABELS = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise', 'contempt']

LABEL_MAP = {
    'anger':     'angry',
    'disgust':   'disgust',
    'fear':      'fear',
    'happiness': 'happy',
    'neutral':   'neutral',
    'sadness':   'sad',
    'surprise':  'surprise',
    'contempt':  'contempt',
}

# FIX 1: Upgraded model for better accuracy
# Options: enet_b0_8_best_afew (fast), enet_b2_8_best_afew (balanced), enet_b2_8_best_vgaf (best)
HSEMOTION_MODEL = 'enet_b0_8_best_afew'  # b2 needs manual download — upgrade later

# FIX 2: Average this many frames per second for smoother readings
FRAMES_TO_AVERAGE = 5


def sanitize(obj):
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize(i) for i in obj]
    elif isinstance(obj, bool):
        return bool(obj)
    elif hasattr(obj, 'item'):
        return obj.item()
    elif isinstance(obj, float):
        return float(obj)
    elif isinstance(obj, int):
        return int(obj)
    return obj


def preprocess_face(face_img):
    """FIX 3: Resize + CLAHE contrast enhancement for better detection in dim lighting."""
    face_img = cv2.resize(face_img, (224, 224))
    lab = cv2.cvtColor(face_img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


class MediaPipeGazeDetector:
    """FIX 4: MediaPipe iris landmarks — far more accurate than OpenCV pupil contours."""

    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def detect_gaze(self, frame) -> dict:
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)

            if not results.multi_face_landmarks:
                return {"eye_contact": False, "gaze_score": 0.0, "eyes_detected": 0}

            lm = results.multi_face_landmarks[0].landmark
            left_score  = self._eye_score(lm, iris=468, inner=133, outer=33)
            right_score = self._eye_score(lm, iris=473, inner=263, outer=362)
            avg = round((left_score + right_score) / 2, 2)

            return {"eye_contact": avg >= 0.55, "gaze_score": avg, "eyes_detected": 2}
        except Exception:
            return {"eye_contact": False, "gaze_score": 0.0, "eyes_detected": 0}

    def _eye_score(self, lm, iris, inner, outer) -> float:
        try:
            eye_w = abs(lm[outer].x - lm[inner].x)
            if eye_w < 1e-6:
                return 0.5
            pos = (lm[iris].x - min(lm[inner].x, lm[outer].x)) / eye_w
            return float(max(0.0, 1.0 - abs(pos - 0.5) * 2))
        except Exception:
            return 0.5


class OpenCVGazeDetector:
    """Fallback when mediapipe is not installed."""

    THRESHOLD = 0.25

    def __init__(self):
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )

    def detect_gaze(self, face_img) -> dict:
        try:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            eyes = self.eye_cascade.detectMultiScale(gray, 1.1, 5, minSize=(20, 20))
            if len(eyes) == 0:
                return {"eye_contact": False, "gaze_score": 0.0, "eyes_detected": 0}

            scores = []
            for (ex, ey, ew, eh) in eyes[:2]:
                roi = gray[ey:ey+eh, ex:ex+ew]
                _, thresh = cv2.threshold(roi, 50, 255, cv2.THRESH_BINARY_INV)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue
                M = cv2.moments(max(contours, key=cv2.contourArea))
                if M["m00"] == 0:
                    continue
                dev = (abs(M["m10"]/M["m00"] - ew/2)/ew + abs(M["m01"]/M["m00"] - eh/2)/eh) / 2
                scores.append(max(0.0, 1.0 - dev / self.THRESHOLD))

            if not scores:
                return {"eye_contact": False, "gaze_score": 0.0, "eyes_detected": len(eyes)}
            avg = round(sum(scores) / len(scores), 2)
            return {"eye_contact": avg >= 0.5, "gaze_score": avg, "eyes_detected": len(eyes)}
        except Exception:
            return {"eye_contact": False, "gaze_score": 0.0, "eyes_detected": 0}


class EmotionTracker:
    def __init__(self):
        self.recognizer  = HSEmotionRecognizer(model_name=HSEMOTION_MODEL)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        if MEDIAPIPE_AVAILABLE:
            self.gaze_detector = MediaPipeGazeDetector()
            print("Using MediaPipe iris tracking")
        else:
            self.gaze_detector = OpenCVGazeDetector()
            print("Using OpenCV fallback gaze detection")

        self.timeline   = []
        self.is_running = False
        self.second     = 0
        self._thread    = None
        self.cap        = None

    def start_tracking(self):
        self.cap        = cv2.VideoCapture(0)
        self.timeline   = []
        self.second     = 0
        self.is_running = True
        self._thread    = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def _get_averaged_emotion(self, frame, faces):
        """FIX 2: Average FRAMES_TO_AVERAGE frames to reduce single-frame noise."""
        all_scores = []
        for _ in range(FRAMES_TO_AVERAGE):
            ret, f = self.cap.read()
            if not ret:
                continue
            try:
                g  = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                fs = self.face_cascade.detectMultiScale(g, 1.1, 4)
                if len(fs) == 0:
                    continue
                x, y, w, h = fs[0]
                face = preprocess_face(f[y:y+h, x:x+w])
                _, scores = self.recognizer.predict_emotions(face, logits=False)
                all_scores.append(scores)
            except Exception:
                continue

        if not all_scores:
            x, y, w, h = faces[0]
            face = preprocess_face(frame[y:y+h, x:x+w])
            _, scores = self.recognizer.predict_emotions(face, logits=False)
            all_scores = [scores]

        avg_scores   = np.mean(all_scores, axis=0)
        dominant_idx = int(np.argmax(avg_scores))
        raw          = EMOTION_LABELS[dominant_idx]
        dominant     = LABEL_MAP.get(raw, raw)
        emotions     = {
            LABEL_MAP.get(lbl, lbl): round(float(s), 2)
            for lbl, s in zip(EMOTION_LABELS, avg_scores)
        }
        return dominant, emotions

    def _sample_loop(self):
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(1)
                continue

            try:
                gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

                if len(faces) > 0:
                    x, y, w, h = faces[0]

                    dominant, emotions = self._get_averaged_emotion(frame, faces)

                    # MediaPipe needs full frame; OpenCV needs face crop
                    gaze = self.gaze_detector.detect_gaze(
                        frame if MEDIAPIPE_AVAILABLE else frame[y:y+h, x:x+w]
                    )

                    negative = sum(emotions.get(k, 0) for k in ('fear','angry','disgust','sad','contempt'))
                    positive = emotions.get('neutral', 0) + emotions.get('happy', 0) * 0.7
                    neutrality_score = round(min(100, max(0, (positive - negative * 0.5 + 0.5) * 100)), 1)

                    self.timeline.append({
                        "second":           self.second,
                        "emotion":          dominant,
                        "confidence":       round(emotions.get(dominant, 0.0), 2),
                        "all":              emotions,
                        "neutrality_score": neutrality_score,
                        "eye_contact":      bool(gaze["eye_contact"]),
                        "gaze_score":       float(gaze["gaze_score"]),
                        "eyes_detected":    int(gaze["eyes_detected"]),
                        "face_detected":    True
                    })
                else:
                    self.timeline.append({
                        "second": self.second, "emotion": None, "confidence": 0.0,
                        "all": {}, "neutrality_score": 0.0, "eye_contact": False,
                        "gaze_score": 0.0, "eyes_detected": 0, "face_detected": False
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
        timeline    = self.timeline
        total       = len(timeline) or 1
        face_frames = [t for t in timeline if t.get("face_detected")]
        face_total  = len(face_frames) or 1

        emotion_counts = {}
        for e in face_frames:
            em = e.get("emotion")
            if em:
                emotion_counts[em] = emotion_counts.get(em, 0) + 1

        dominant    = max(emotion_counts, key=emotion_counts.get) if emotion_counts else "neutral"
        nervous_pct = ((emotion_counts.get("fear",0) + emotion_counts.get("sad",0)) / face_total) * 100
        happy_pct   = (emotion_counts.get("happy",   0) / face_total) * 100
        neutral_pct = (emotion_counts.get("neutral", 0) / face_total) * 100
        angry_pct   = (emotion_counts.get("angry",   0) / face_total) * 100
        conf_score  = min(100, max(0, round(happy_pct*0.8 + neutral_pct*0.5 - nervous_pct*0.5 + 50)))

        avg_neutrality  = round(sum(t["neutrality_score"] for t in face_frames) / face_total, 1) if face_frames else 0.0
        eye_ok_frames   = [t for t in face_frames if t.get("eye_contact")]
        eye_contact_pct = round(len(eye_ok_frames) / face_total * 100, 1)
        avg_gaze        = round(sum(t.get("gaze_score",0) for t in face_frames) / face_total, 2)

        if eye_contact_pct >= 75:   quality = "excellent"
        elif eye_contact_pct >= 50: quality = "good"
        elif eye_contact_pct >= 25: quality = "needs_improvement"
        else:                       quality = "poor"

        delivery_score = round(conf_score*0.5 + avg_neutrality*0.3 + eye_contact_pct*0.2, 1)

        result = {
            "part": part, "question": question,
            "duration_seconds": total, "face_detected_seconds": len(face_frames),
            "gaze_method":   "mediapipe" if MEDIAPIPE_AVAILABLE else "opencv",
            "emotion_model": HSEMOTION_MODEL,
            "emotion_timeline": timeline,
            "summary": {
                "dominant_emotion":         dominant,
                "confidence_score":         conf_score,
                "nervous_percentage":       round(nervous_pct, 1),
                "happy_percentage":         round(happy_pct, 1),
                "neutral_percentage":       round(neutral_pct, 1),
                "angry_percentage":         round(angry_pct, 1),
                "emotion_breakdown":        emotion_counts,
                "avg_neutrality_score":     avg_neutrality,
                "eye_contact_percentage":   eye_contact_pct,
                "eye_contact_quality":      quality,
                "avg_gaze_score":           avg_gaze,
                "delivery_score":           delivery_score,
                "nervous_moments":          [f"second {t['second']}" for t in face_frames if t.get("emotion") in ("fear","sad") and t.get("confidence",0) > 0.5][:5],
                "lost_eye_contact_moments": [f"second {t['second']}" for t in face_frames if not t.get("eye_contact") and t.get("eyes_detected",0) > 0][:5],
                "face_absent_seconds":      [t["second"] for t in timeline if not t.get("face_detected")][:5],
            }
        }
        return sanitize(result)


tracker = EmotionTracker()


@router.post("/emotion/start")
async def start_emotion_tracking():
    tracker.start_tracking()
    return {"status": "tracking started"}


@router.post("/emotion/stop")
async def stop_emotion_tracking(question: str = "", part: int = 1):
    result = tracker.stop_tracking(question=question, part=part)
    return JSONResponse(content=result)