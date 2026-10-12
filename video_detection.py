import cv2
import time
import numpy as np
from collections import deque, OrderedDict
from ultralytics import YOLO
from scipy.spatial import distance as dist
from scipy.stats import poisson

# â”€â”€ CONFIG â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
MODEL_PATH       = "models/best.pt"
VIDEO_PATH       = "road.mp4"

CONF_THRESH_SMALL  = 0.30
CONF_THRESH_MEDIUM = 0.40
CONF_THRESH_LARGE  = 0.45

SKIP_FRAMES      = 3
MAX_DISAPPEARED  = 20
MIN_CONFIRM      = 3
CENTROID_DIST    = 80
AREA_SMOOTH_WIN  = 8

POTHOLE_CLASS_NAME = "pothole"   # set None to accept all classes

SEVERITY_COLORS = {
    "Small":  (0, 220, 80),
    "Medium": (0, 180, 255),
    "Large":  (0, 60, 230),
}

SEVERITY_WEIGHTS = {"Small": 1, "Medium": 2, "Large": 3}
SCORE_GOOD     = 1.5
SCORE_MODERATE = 2.5

HEATMAP_DECAY = 0.995

# â”€â”€ CENTROID TRACKER â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class CentroidTracker:
    def __init__(self):
        self.next_id      = 0
        self.objects      = OrderedDict()
        self.boxes        = OrderedDict()
        self.disappeared  = OrderedDict()
        self.hits         = OrderedDict()
        self.categories   = OrderedDict()
        self.area_history = OrderedDict()
        self.confirmed    = set()
        self.counted      = set()

    def _stable_severity(self, tid, raw_area, img_area):
        if tid not in self.area_history:
            self.area_history[tid] = deque(maxlen=AREA_SMOOTH_WIN)
        self.area_history[tid].append(raw_area)
        avg_ratio = np.mean(self.area_history[tid]) / img_area
        if avg_ratio < 0.01:
            return "Small"
        elif avg_ratio < 0.03:
            return "Medium"
        return "Large"

    def register(self, centroid, box, category):
        self.objects[self.next_id]     = centroid
        self.boxes[self.next_id]       = box
        self.disappeared[self.next_id] = 0
        self.hits[self.next_id]        = 1
        self.categories[self.next_id]  = category
        self.next_id += 1

    def deregister(self, tid):
        for d in (self.objects, self.boxes, self.disappeared,
                  self.hits, self.categories, self.area_history):
            d.pop(tid, None)

    def update(self, detections, img_area):
        if not detections:
            for tid in list(self.disappeared):
                self.disappeared[tid] += 1
                if self.disappeared[tid] > MAX_DISAPPEARED:
                    self.deregister(tid)
            return []

        input_centroids = []
        for d in detections:
            x1, y1, x2, y2 = d["box"]
            input_centroids.append(((x1+x2)//2, (y1+y2)//2))

        if not self.objects:
            for i, d in enumerate(detections):
                raw_area = (d["box"][2]-d["box"][0]) * (d["box"][3]-d["box"][1])
                cat = self._stable_severity(self.next_id, raw_area, img_area)
                self.register(input_centroids[i], d["box"], cat)
            results = []
            keys = list(self.objects.keys())
            for i, d in enumerate(detections):
                tid = keys[i] if i < len(keys) else 0
                d.update(track_id=tid, confirmed=False, is_new_count=False,
                         hits=1, category=self.categories.get(tid, d["category"]))
                results.append(d)
            return results

        existing_ids       = list(self.objects.keys())
        existing_centroids = list(self.objects.values())
        D    = dist.cdist(np.array(existing_centroids), np.array(input_centroids))
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows, used_cols, matched = set(), set(), {}
        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] > CENTROID_DIST:
                continue
            tid = existing_ids[row]
            matched[col] = tid
            used_rows.add(row)
            used_cols.add(col)

        for col, tid in matched.items():
            d = detections[col]
            cx, cy = input_centroids[col]
            px, py = self.objects[tid]
            self.objects[tid]     = (int(0.7*px+0.3*cx), int(0.7*py+0.3*cy))
            self.boxes[tid]       = d["box"]
            self.disappeared[tid] = 0
            self.hits[tid]       += 1
            raw_area = (d["box"][2]-d["box"][0]) * (d["box"][3]-d["box"][1])
            self.categories[tid] = self._stable_severity(tid, raw_area, img_area)

        for col in range(len(detections)):
            if col not in matched:
                raw_area = (detections[col]["box"][2]-detections[col]["box"][0]) * \
                           (detections[col]["box"][3]-detections[col]["box"][1])
                cat = self._stable_severity(self.next_id, raw_area, img_area)
                self.register(input_centroids[col], detections[col]["box"], cat)

        for row, tid in enumerate(existing_ids):
            if row not in used_rows:
                self.disappeared[tid] += 1
                if self.disappeared[tid] > MAX_DISAPPEARED:
                    self.deregister(tid)

        results = []
        new_keys = list(self.objects.keys())
        for col, d in enumerate(detections):
            if col in matched:
                tid = matched[col]
            else:
                unmatched = [c for c in range(len(detections)) if c not in matched]
                idx = unmatched.index(col) if col in unmatched else 0
                tid = new_keys[-(len(unmatched)-idx)] if new_keys else 0

            is_confirmed = self.hits.get(tid, 0) >= MIN_CONFIRM
            is_new_count = is_confirmed and tid not in self.counted
            if is_new_count:
                self.counted.add(tid)
                self.confirmed.add(tid)

            d.update(track_id=tid, confirmed=is_confirmed,
                     is_new_count=is_new_count, hits=self.hits.get(tid, 0),
                     category=self.categories.get(tid, d["category"]))
            results.append(d)
        return results

# â”€â”€ ROAD CONDITION SCORE â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def road_condition_score(small, medium, large):
    total = small + medium + large
    if total == 0:
        return 0.0, "N/A", (180, 180, 180)
    score = (SEVERITY_WEIGHTS["Small"]*small +
             SEVERITY_WEIGHTS["Medium"]*medium +
             SEVERITY_WEIGHTS["Large"]*large) / total
    if score < SCORE_GOOD:
        label, color = "Good",     (0, 220, 80)
    elif score <= SCORE_MODERATE:
        label, color = "Moderate", (0, 180, 255)
    else:
        label, color = "Poor",     (0, 60, 230)
    return score, label, color

# â”€â”€ STATISTICS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def compute_stats(area_list):
    if not area_list:
        return 0.0, 0.0
    arr = np.array(area_list, dtype=np.float64)
    return float(np.mean(arr)), float(np.var(arr))

def poisson_prob(lam, k):
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return float(poisson.pmf(k, lam))

# â”€â”€ DRAW HELPERS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def draw_rounded_rect(img, pt1, pt2, color, thickness=2, r=8):
    x1, y1 = pt1; x2, y2 = pt2
    cv2.line(img, (x1+r, y1), (x2-r, y1), color, thickness)
    cv2.line(img, (x1+r, y2), (x2-r, y2), color, thickness)
    cv2.line(img, (x1, y1+r), (x1, y2-r), color, thickness)
    cv2.line(img, (x2, y1+r), (x2, y2-r), color, thickness)
    cv2.ellipse(img, (x1+r, y1+r), (r,r), 180,  0, 90, color, thickness)
    cv2.ellipse(img, (x2-r, y1+r), (r,r), 270,  0, 90, color, thickness)
    cv2.ellipse(img, (x1+r, y2-r), (r,r),  90,  0, 90, color, thickness)
    cv2.ellipse(img, (x2-r, y2-r), (r,r),   0,  0, 90, color, thickness)

def draw_label_pill(img, text, origin, color, font_scale=0.50):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), bl = cv2.getTextSize(text, font, font_scale, 1)
    x, y = origin; pad = 4
    cv2.rectangle(img, (x-pad, y-th-pad), (x+tw+pad, y+bl+pad), color, -1)
    cv2.putText(img, text, (x, y), font, font_scale, (255,255,255), 1, cv2.LINE_AA)

def draw_overlay_panel(img, lines, start_y=30, x=18):
    font = cv2.FONT_HERSHEY_SIMPLEX; line_h = 32
    panel_h = line_h * len(lines) + 16
    max_w   = max(cv2.getTextSize(t, font, sc, 2)[0][0] for t,sc,_ in lines) + 36
    ov = img.copy()
    cv2.rectangle(ov, (x-8, start_y-20), (x+max_w, start_y+panel_h), (20,20,20), -1)
    cv2.addWeighted(ov, 0.55, img, 0.45, 0, img)
    for i, (text, scale, color) in enumerate(lines):
        cv2.putText(img, text, (x, start_y+i*line_h),
                    font, scale, color, 2, cv2.LINE_AA)

def draw_condition_badge(img, score, label, color, w):
    badge_x = w - 235
    badge_y = 10
    ov = img.copy()
    cv2.rectangle(ov, (badge_x, badge_y), (w-10, badge_y+62), (20,20,20), -1)
    cv2.addWeighted(ov, 0.60, img, 0.40, 0, img)
    cv2.putText(img, "Road Condition", (badge_x+8, badge_y+20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160,160,160), 1, cv2.LINE_AA)
    cv2.putText(img, f"{label}  ({score:.2f})", (badge_x+8, badge_y+50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA)

def draw_mini_bar(img, small, medium, large):
    x = 18; y = img.shape[0] - 90
    total = small + medium + large or 1
    bw, bh = 180, 14
    items = [("S",small,(0,220,80)), ("M",medium,(0,180,255)), ("L",large,(0,60,230))]
    cv2.putText(img, "Severity breakdown", (x, y-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160,160,160), 1, cv2.LINE_AA)
    cx = x
    for _, count, color in items:
        sw = int(bw * count / total)
        if sw > 0:
            cv2.rectangle(img, (cx, y), (cx+sw, y+bh), color, -1)
        cx += sw
    cv2.rectangle(img, (x, y), (x+bw, y+bh), (100,100,100), 1)
    for i, (tag, count, color) in enumerate(items):
        lx = x + i*62
        cv2.rectangle(img, (lx, y+bh+6), (lx+10, y+bh+16), color, -1)
        cv2.putText(img, f"{tag}:{count}", (lx+13, y+bh+16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200,200,200), 1, cv2.LINE_AA)

def draw_dashed_rect(img, x1, y1, x2, y2, color=(180,180,180), gap=10):
    for i in range(x1, x2, gap):
        cv2.line(img, (i, y1), (min(i+gap//2, x2), y1), color, 1)
        cv2.line(img, (i, y2), (min(i+gap//2, x2), y2), color, 1)
    for i in range(y1, y2, gap):
        cv2.line(img, (x1, i), (x1, min(i+gap//2, y2)), color, 1)
        cv2.line(img, (x2, i), (x2, min(i+gap//2, y2)), color, 1)

# â”€â”€ MAIN â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
model = YOLO(MODEL_PATH)
cap   = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open: {VIDEO_PATH}")

orig_w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_fps = cap.get(cv2.CAP_PROP_FPS)
frame_delay = int(1000 / video_fps)
print(f"Video: {orig_w}x{orig_h} @ {video_fps:.1f} fps  â†’  delay {frame_delay}ms/frame")

pothole_class_ids = None
if POTHOLE_CLASS_NAME and hasattr(model, "names"):
    pothole_class_ids = {i for i, n in model.names.items()
                         if POTHOLE_CLASS_NAME.lower() in n.lower()}
    print(f"Filtering to class IDs: {pothole_class_ids} ({POTHOLE_CLASS_NAME})")

heatmap_acc  = np.zeros((orig_h, orig_w), dtype=np.float32)
tracker      = CentroidTracker()
box_memory   = {}

mode         = "count"
total_count  = 0
small = medium = large = 0

confirmed_areas    = []
active_potholes_per_frame = []

frame_idx    = 0
paused       = False
last_tracked = []

fps_buf = deque(maxlen=30)
t_prev  = time.time()
MODES   = {ord('1'):"count",   ord('2'):"severity",
           ord('3'):"stats",   ord('4'):"heatmap",
           ord('5'):"poisson"}

print("Controls: 1=Count  2=Severity  3=Stats  4=Heatmap  5=Poisson  P=Pause  R=Reset  ESC=Quit")

while cap.isOpened():
    t_frame_start = time.time()

    if not paused:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

    t_now = time.time()
    fps_buf.append(1.0 / max(t_now - t_prev, 1e-6))
    t_prev = t_now
    fps    = np.mean(fps_buf)

    h, w     = frame.shape[:2]
    img_area = w * h

    if not paused:
        heatmap_acc *= HEATMAP_DECAY

    # â”€â”€ INFERENCE â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if not paused and frame_idx % SKIP_FRAMES == 0:
        results = model(frame, verbose=False)
        raw = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if pothole_class_ids and cls_id not in pothole_class_ids:
                    continue

                conf  = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area  = (x2-x1) * (y2-y1)
                ratio = area / img_area

                thresh = (CONF_THRESH_SMALL if ratio < 0.01 else
                          CONF_THRESH_MEDIUM if ratio < 0.03 else
                          CONF_THRESH_LARGE)
                if conf < thresh:
                    continue

                cat = "Small" if ratio < 0.01 else "Medium" if ratio < 0.03 else "Large"
                raw.append({"box":[x1,y1,x2,y2], "category":cat, "conf":conf})

        last_tracked = tracker.update(raw, img_area)
        active_potholes_per_frame.append(len(tracker.objects))

        new_this_frame = 0
        for det in last_tracked:
            tid = det["track_id"]
            b = det["box"]
            if tid in box_memory:
                pb = box_memory[tid]
                smoothed = [int(0.6*pb[i]+0.4*b[i]) for i in range(4)]
            else:
                smoothed = b
            box_memory[tid] = smoothed
            det["box"]      = smoothed

            if det["is_new_count"]:
                total_count += 1
                cat = det["category"]
                if   cat == "Small":  small  += 1
                elif cat == "Medium": medium += 1
                else:                 large  += 1
                x1,y1,x2,y2 = smoothed
                confirmed_areas.append((x2-x1)*(y2-y1))
                new_this_frame += 1
                heat_val = {"Small":1,"Medium":2,"Large":4}[cat]
                cv2.rectangle(heatmap_acc,(x1,y1),(x2,y2),heat_val,-1)

       

    live_ids = set(tracker.objects.keys())
    for tid in list(box_memory):
        if tid not in live_ids:
            del box_memory[tid]

    # â”€â”€ DRAW DETECTIONS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    for det in last_tracked:
        tid = det["track_id"]
        if tid not in tracker.objects:
            continue
        x1, y1, x2, y2 = det.get("box",[0,0,0,0])
        cat   = det["category"]
        conf  = det["conf"]
        hits  = det.get("hits",0)
        color = SEVERITY_COLORS[cat]

        if det.get("confirmed"):
            draw_rounded_rect(frame, (x1,y1), (x2,y2), color, 2)
            draw_label_pill(frame, f"#{tid} {cat} | {conf:.2f}", (x1,y1-6), color)
        else:
            draw_dashed_rect(frame, x1, y1, x2, y2)
            cv2.putText(frame, f"{hits}/{MIN_CONFIRM}", (x1,y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180,180,180), 1, cv2.LINE_AA)

    # â”€â”€ LIVE COMPUTED METRICS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    score, cond_label, cond_color = road_condition_score(small, medium, large)
    mean_area, var_area = compute_stats(confirmed_areas)
    lam = float(np.mean(active_potholes_per_frame)) if active_potholes_per_frame else 0.0

    # â”€â”€ MODE OVERLAYS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if mode == "count":
        draw_overlay_panel(frame, [
            (f"Unique potholes : {total_count}",             0.75, (255,255,255)),
            (f"Active tracks   : {len(tracker.objects)}",    0.50, (160,160,160)),
            (f"Frames scanned  : {frame_idx}",               0.50, (160,160,160)),
        ])

    elif mode == "severity":
        draw_overlay_panel(frame, [
            (f"Small  (<1% area) : {small}",      0.60, SEVERITY_COLORS["Small"]),
            (f"Medium (1-3% area): {medium}",     0.60, SEVERITY_COLORS["Medium"]),
            (f"Large  (>3% area) : {large}",      0.60, SEVERITY_COLORS["Large"]),
            (f"Total unique      : {total_count}",0.52, (220,220,220)),
        ])
        draw_mini_bar(frame, small, medium, large)

    elif mode == "stats":
        draw_overlay_panel(frame, [
            (f"Mean pothole area : {mean_area:.0f} px2", 0.58, (220,220,220)),
            (f"Variance          : {var_area:.0f} px2",  0.58, (220,220,220)),
            (f"Std deviation     : {np.sqrt(max(var_area,0)):.0f} px", 0.54, (180,180,180)),
            (f"Poisson  lambda   : {lam:.4f} /frame",    0.54, (255,220,80)),
        ])

    elif mode == "heatmap":
        if heatmap_acc.max() > 0:
            norm = cv2.normalize(heatmap_acc, None, 0, 255, cv2.NORM_MINMAX)
            heat = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_JET)
            frame = cv2.addWeighted(frame, 0.6, heat, 0.4, 0)
        draw_overlay_panel(frame,[("Heatmap - pothole density (decayed)",0.52,(255,255,255))])

    elif mode == "poisson":
        p0 = poisson_prob(lam, 0)
        p1 = poisson_prob(lam, 1)
        p2 = poisson_prob(lam, 2)
        draw_overlay_panel(frame, [
            (f"Poisson model  lambda = {lam:.4f}", 0.58, (255,220,80)),
            (f"P(X=0) = {p0:.4f}",                 0.52, (200,200,200)),
            (f"P(X=1) = {p1:.4f}",                 0.52, (200,200,200)),
            (f"P(X=2) = {p2:.4f}",                 0.52, (200,200,200)),
            (f"P(X>=3)= {1-p0-p1-p2:.4f}",         0.52, (200,200,200)),
        ])

    # Always-visible widgets
    draw_condition_badge(frame, score, cond_label, cond_color, w)
    if mode != "severity":
        draw_mini_bar(frame, small, medium, large)

    cv2.putText(frame, f"FPS:{fps:.1f}", (w-100, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (160,160,160), 1, cv2.LINE_AA)
    cv2.putText(frame,
                "1:Count 2:Severity 3:Stats 4:Heatmap 5:Poisson  P:Pause R:Reset ESC:Quit",
                (18, h-14), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (120,120,120), 1, cv2.LINE_AA)
    if paused:
        cv2.putText(frame, "PAUSED", (w//2-60, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0,120,255), 3, cv2.LINE_AA)

    cv2.imshow("Pothole Detector", frame)

    t_inf_ms = int((time.time()-t_frame_start)*1000)
    key = cv2.waitKey(max(1, frame_delay-t_inf_ms)) & 0xFF

    if key == 27:
        break
    elif key in MODES:
        mode = MODES[key]
    elif key in (ord('p'), ord('P')):
        paused = not paused
    elif key in (ord('r'), ord('R')):
        tracker = CentroidTracker()
        box_memory.clear()
        heatmap_acc[:] = 0
        total_count = small = medium = large = frame_idx = 0
        confirmed_areas.clear()
        active_potholes_per_frame.clear()
        last_tracked = []
        print("Reset.")

cap.release()
cv2.destroyAllWindows()

# â”€â”€ FINAL REPORT â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
score, cond_label, _ = road_condition_score(small, medium, large)
mean_area, var_area  = compute_stats(confirmed_areas)
lam = float(np.mean(active_potholes_per_frame)) if active_potholes_per_frame else 0.0

print(f"\nâ”€â”€ Detection Summary â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
print(f"  Unique potholes           : {total_count}")
print(f"  Small / Medium / Large    : {small} / {medium} / {large}")
print(f"  Frames scanned            : {frame_idx}")
print(f"  Video FPS                 : {video_fps:.1f}")

print(f"\nâ”€â”€ Road Condition Score â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
print(f"  Score  (weighted avg)     : {score:.3f}")
print(f"  Condition                 : {cond_label}")
print(f"  Scale : Good < 1.5 | 1.5 <= Moderate <= 2.5 | Poor > 2.5")

print(f"\nâ”€â”€ Statistical Analysis â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
print(f"  Mean pothole area         : {mean_area:.2f} px2")
print(f"  Variance                  : {var_area:.2f} px2")
print(f"  Std deviation             : {np.sqrt(max(var_area,0)):.2f} px")

print(f"\nâ”€â”€ Poisson Distribution (lambda = active avg potholes per frame) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
print(f"  lambda                     : {lam:.4f}")
for k in range(5):
    print(f"  P(X={k})                   : {poisson_prob(lam,k):.6f}")
print(f"  P(X>=3)                   : {1-sum(poisson_prob(lam,k) for k in range(3)):.6f}")
