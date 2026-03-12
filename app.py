import json
import os
import shutil
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Tuple

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flask import Flask, jsonify, make_response, render_template, request, send_file
import cv2
import numpy as np
from ultralytics import YOLO

from path_planning import plan_shortest_path

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(APP_ROOT, "best.pt")
UPLOAD_DIR = os.path.join(APP_ROOT, "static", "uploads")
RESULT_DIR = os.path.join(APP_ROOT, "static", "results")
HISTORY_DIR = os.path.join(APP_ROOT, "data")
HISTORY_FILE = os.path.join(HISTORY_DIR, "history.json")
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
MAX_FILES = 9
HISTORY_KEEP_SECONDS = 7 * 24 * 60 * 60
HISTORY_PAGE_SIZE = 3

MODEL = None


def ensure_dirs() -> None:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(HISTORY_DIR, exist_ok=True)


def allowed_file(filename: str) -> bool:
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_EXTENSIONS


def load_history() -> List[Dict[str, Any]]:
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as handle:
            data = json.load(handle)
            if isinstance(data, list):
                for batch in data:
                    if "records" not in batch and "items" in batch:
                        batch["records"] = batch.pop("items")
                return data
    except (OSError, json.JSONDecodeError):
        return []
    return []


def save_history(history: List[Dict[str, Any]]) -> None:
    with open(HISTORY_FILE, "w", encoding="utf-8") as handle:
        json.dump(history, handle, ensure_ascii=False, indent=2)


def cleanup_history(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    now = time.time()
    kept: List[Dict[str, Any]] = []
    for batch in history:
        created_ts = batch.get("created_ts", 0)
        if created_ts and now - float(created_ts) <= HISTORY_KEEP_SECONDS:
            kept.append(batch)
            continue

        upload_dir = batch.get("upload_dir")
        result_dir = batch.get("result_dir")
        for dir_path in (upload_dir, result_dir):
            if dir_path and os.path.isdir(dir_path):
                shutil.rmtree(dir_path, ignore_errors=True)

    return kept


def get_model() -> YOLO:
    global MODEL
    if MODEL is None:
        if not os.path.exists(WEIGHTS_PATH):
            raise FileNotFoundError(f"模型文件不存在: {WEIGHTS_PATH}")
        MODEL = YOLO(WEIGHTS_PATH).to("cpu")
    return MODEL


def draw_detections_and_path(
    image: np.ndarray,
    boxes: List[Tuple[float, float, float, float]],
    confidences: List[float],
) -> np.ndarray:
    output = image.copy()
    points: List[Tuple[int, int]] = []

    for (x1, y1, x2, y2), conf in zip(boxes, confidences):
        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))
        cv2.rectangle(output, p1, p2, (0, 255, 0), 2)
        cv2.putText(
            output,
            f"{conf:.2f}",
            (p1[0], max(p1[1] - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        points.append((cx, cy))
        cv2.circle(output, (cx, cy), 4, (0, 0, 255), -1)

    if points:
        h, w = output.shape[:2]
        shortest_path, _ = plan_shortest_path(points, w, h)

        for i in range(len(shortest_path) - 1):
            pt1 = shortest_path[i]
            pt2 = shortest_path[i + 1]
            cv2.line(output, pt1, pt2, (0, 255, 255), 2)

        if shortest_path:
            cv2.circle(output, shortest_path[0], 8, (255, 0, 0), -1)
            cv2.putText(
                output,
                "Start",
                (shortest_path[0][0] + 8, shortest_path[0][1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
            )

        for i, pt in enumerate(shortest_path[1:]):
            cv2.putText(
                output,
                str(i + 1),
                (pt[0] + 4, pt[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )

    return output


def process_image(image_path: str, output_path: str) -> Tuple[int, float, float, List[float]]:
    model = get_model()
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("无法读取图片")

    results = model.predict(
        image_path,
        conf=0.25,
        imgsz=256,
        verbose=False,
        iou=0.45,
        device="cpu",
        show_labels=True,
        show_conf=True,
    )
    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        boxes = []
        confidences = []
    else:
        boxes = result.boxes.xyxy.cpu().numpy().tolist()
        confidences = result.boxes.conf.cpu().numpy().tolist()

    avg_conf = float(np.mean(confidences)) if confidences else 0.0
    max_conf = float(np.max(confidences)) if confidences else 0.0

    output_img = draw_detections_and_path(image, boxes, confidences)
    cv2.imwrite(output_path, output_img)

    return len(boxes), avg_conf, max_conf, confidences


def process_image_for_api(image_path: str) -> Dict[str, Any]:
    model = get_model()
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("??????")

    results = model.predict(
        image_path,
        conf=0.25,
        imgsz=256,
        verbose=False,
        iou=0.45,
        device="cpu",
        show_labels=True,
        show_conf=True,
    )
    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        boxes = []
        confidences = []
    else:
        boxes = result.boxes.xyxy.cpu().numpy().tolist()
        confidences = result.boxes.conf.cpu().numpy().tolist()

    h, w = image.shape[:2]
    points = []
    for (x1, y1, x2, y2), conf in zip(boxes, confidences):
        cx, cy = float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)
        points.append(
            {
                "x": cx,
                "y": cy,
                "nx": float(np.clip(cx / max(w, 1), 0.0, 1.0)),
                "ny": float(np.clip(cy / max(h, 1), 0.0, 1.0)),
                "conf": float(conf),
            }
        )

    avg_conf = float(np.mean(confidences)) if confidences else 0.0
    max_conf = float(np.max(confidences)) if confidences else 0.0

    api_result_dir = os.path.join(RESULT_DIR, "api")
    os.makedirs(api_result_dir, exist_ok=True)
    result_name = f"api_{uuid.uuid4().hex[:10]}.jpg"
    result_path = os.path.join(api_result_dir, result_name)
    output_img = draw_detections_and_path(image, boxes, confidences)
    cv2.imwrite(result_path, output_img)

    return {
        "detections": len(boxes),
        "avg_conf": avg_conf,
        "max_conf": max_conf,
        "image_width": w,
        "image_height": h,
        "points": points,
        "result_image_url": f"/static/results/api/{result_name}",
    }


def _cors_json(payload: Dict[str, Any], status: int = 200):
    resp = make_response(jsonify(payload), status)
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024


@app.route("/", methods=["GET"])
def index():
    welcome_path = os.path.join(APP_ROOT, "welcome.html")
    if not os.path.exists(welcome_path):
        return "welcome page not found", 404
    return send_file(welcome_path)


@app.route("/detect", methods=["GET", "POST"])
def detect():
    ensure_dirs()
    history = cleanup_history(load_history())
    save_history(history)

    if request.method == "POST":
        files = request.files.getlist("images")
        files = [f for f in files if f and f.filename]

        if not files:
            return render_template("index.html", error="请至少上传 1 张图片。")

        if len(files) > MAX_FILES:
            return render_template("index.html", error="最多只能上传 9 张图片。")

        batch_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        batch_upload_dir = os.path.join(UPLOAD_DIR, batch_id)
        batch_result_dir = os.path.join(RESULT_DIR, batch_id)
        os.makedirs(batch_upload_dir, exist_ok=True)
        os.makedirs(batch_result_dir, exist_ok=True)

        results_payload = []
        for file in files:
            if not allowed_file(file.filename):
                return render_template("index.html", error=f"不支持的文件格式: {file.filename}")

            safe_name = os.path.basename(file.filename)
            upload_path = os.path.join(batch_upload_dir, safe_name)
            result_path = os.path.join(batch_result_dir, safe_name)

            file.save(upload_path)

            try:
                count, avg_conf, max_conf, confidences = process_image(upload_path, result_path)
            except Exception as exc:
                return render_template("index.html", error=f"处理 {safe_name} 失败: {exc}")

            results_payload.append(
                {
                    "filename": safe_name,
                    "detections": count,
                    "avg_conf": avg_conf,
                    "max_conf": max_conf,
                    "confidences": confidences,
                    "upload_url": f"/static/uploads/{batch_id}/{safe_name}",
                    "result_url": f"/static/results/{batch_id}/{safe_name}",
                }
            )
        batch_record = {
            "batch_id": batch_id,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "created_ts": time.time(),
            "upload_dir": batch_upload_dir,
            "result_dir": batch_result_dir,
            "records": results_payload,
        }

        history.insert(0, batch_record)
        save_history(history)

        return render_template(
            "index.html",
            results=results_payload,
            batch_id=batch_id,
        )

    return render_template("index.html")


@app.route("/api/detect", methods=["POST", "OPTIONS"])
def api_detect():
    ensure_dirs()
    if request.method == "OPTIONS":
        return _cors_json({"ok": True}, 204)

    file = request.files.get("image")
    if not file or not file.filename:
        return _cors_json({"ok": False, "error": "???????(image)"}, 400)

    if not allowed_file(file.filename):
        return _cors_json({"ok": False, "error": f"????????: {file.filename}"}, 400)

    safe_name = os.path.basename(file.filename)
    temp_name = f"api_{uuid.uuid4().hex[:10]}_{safe_name}"
    upload_path = os.path.join(UPLOAD_DIR, temp_name)
    file.save(upload_path)

    try:
        data = process_image_for_api(upload_path)
    except Exception as exc:
        return _cors_json({"ok": False, "error": f"????: {exc}"}, 500)

    return _cors_json({"ok": True, **data})


@app.route("/sim", methods=["GET"])
def sim_page():
    sim_path = os.path.join(APP_ROOT, "index.html")
    if not os.path.exists(sim_path):
        return "sim page not found", 404
    return send_file(sim_path)


@app.route("/history", methods=["GET"])
def history_page():
    ensure_dirs()
    history = cleanup_history(load_history())
    save_history(history)
    page = max(1, int(request.args.get("page", 1)))
    total_pages = max(1, (len(history) + HISTORY_PAGE_SIZE - 1) // HISTORY_PAGE_SIZE)
    start = (page - 1) * HISTORY_PAGE_SIZE
    end = start + HISTORY_PAGE_SIZE
    paged_history = history[start:end]
    return render_template(
        "history.html",
        history_batches=paged_history,
        history_page=page,
        history_pages=total_pages,
    )


if __name__ == "__main__":
    ensure_dirs()
    app.run(host="0.0.0.0", port=5000, debug=True)
