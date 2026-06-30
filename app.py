# -*- coding: utf-8 -*-
"""TaThip Desktop App — pywebview + SQLite + OCR + ประมวลผลวิดีโอ (background)
รัน:  python app.py
API ฝั่ง JS เรียกผ่าน window.pywebview.api.<name>()
"""
import os, sys, base64, json, sqlite3, re, time, threading
from datetime import datetime, timedelta
import webview
import cv2

HERE = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
INDEX = os.path.join(HERE, "index.html")
APPDIR = os.path.dirname(os.path.abspath(__file__))     # โฟลเดอร์เขียนได้จริง
DB = os.path.join(APPDIR, "tathip.db")
IMGDIR = os.path.join(APPDIR, "img"); os.makedirs(IMGDIR, exist_ok=True)
STORE = os.path.join(APPDIR, "tracks_store.json")        # เก็บ track ต่อกล้อง (มี emb)
TRACKS_JS = os.path.join(APPDIR, "tracks_data.js")       # ไฟล์ที่ UI โหลด

# ---- pipeline config ----
# Re-ID weights/extractor (BDI_hackathon). ตั้งทับได้ด้วย env TATHIP_BDI
BDI = os.environ.get("TATHIP_BDI", r"D:/software_engineer/project/BDI_hackathon/data/traffic/pipeline")
PROC_MINUTES = 5            # ประมวลผลกี่นาทีจากต้นคลิป (ปรับได้)
COCO = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
TH_CLS = {"car": "เก๋ง", "truck": "รถบรรทุก", "bus": "รถบัส", "motorcycle": "มอเตอร์ไซค์"}
GROUP_TH = 0.66
DEFAULT_DAY = "2026-04-05"
import numpy as np
PROGRESS = {}     # cam_id -> {pct, stage, done, running, n, total, data, error}
# ประมวลผลทีละกล้อง: บังคับใน start_processing (โมเดล/ByteTrack เป็น state เดียว ห้ามรันพร้อมกัน)

def _device():
    """GPU ถ้ามี ไม่งั้น CPU — กันแอปล้มบนเครื่องไม่มี CUDA"""
    try:
        import torch
        return 0 if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

# ยืดหยุ่น: anchor ปี 20xx + ยอมให้ตัวคั่นเป็นอะไรก็ได้/ไม่มี (OCR มักอ่าน : หาย/เลขติดกัน)
TS_RE = re.compile(r"(20\d{2})\D{0,2}(\d{2})\D{0,2}(\d{2})\D{0,4}(\d{2})\D{0,2}(\d{2})\D{0,2}(\d{2})")


def _hms(sec):
    sec = int(sec)
    return f"{sec//3600:02d}:{sec%3600//60:02d}:{sec%60:02d}"


# ---------------- DB ----------------
def db():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    return c

def init_db():
    with db() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS cameras(
            id TEXT PRIMARY KEY, name TEXT, lat REAL, lon REAL,
            video_path TEXT, start_time TEXT, roi TEXT, ts_box TEXT, created TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS saved_paths(
            id TEXT PRIMARY KEY, name TEXT, note TEXT, created TEXT, sel TEXT)""")


# ---------------- pipeline (lazy) : YOLOv8 COCO + ByteTrack + VeRi Re-ID ----------------
_YOLO = None; _EXT = None
def _yolo():
    global _YOLO
    if _YOLO is None:
        from ultralytics import YOLO
        _YOLO = YOLO(os.path.join(BDI, "yolov8s.pt") if os.path.exists(os.path.join(BDI,"yolov8s.pt")) else "yolov8s.pt")
    return _YOLO
def _ext():
    global _EXT
    if _EXT is None:
        ckpt = os.path.join(BDI, "models", "veri_sbs_R50-ibn.pth")
        if not os.path.exists(os.path.join(BDI, "veri_extractor.py")) or not os.path.exists(ckpt):
            raise RuntimeError(
                "ไม่พบ Re-ID (veri_extractor.py + veri_sbs_R50-ibn.pth) ที่ " + BDI +
                " — ตั้ง path ด้วย env TATHIP_BDI")
        sys.path.insert(0, BDI); import veri_extractor as V
        _EXT = V.VeRiExtractor(ckpt=ckpt).to(V.DEVICE)
    return _EXT

def _color(im):
    h, w = im.shape[:2]
    if h < 6 or w < 6: return "ไม่ทราบ", "#888888"
    cen = im[h//4:h*3//4, w//4:w*3//4]
    bgr = np.median(cen.reshape(-1, 3), axis=0)
    hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]
    H, S, Vv = int(hsv[0]), int(hsv[1]), int(hsv[2])
    if Vv < 60: n = "ดำ"
    elif S < 40 and Vv > 175: n = "ขาว"
    elif S < 55: n = "เทา/เงิน"
    elif H < 10 or H >= 170: n = "แดง"
    elif H < 22: n = "ส้ม"
    elif H < 34: n = "เหลือง"
    elif H < 85: n = "เขียว"
    elif H < 130: n = "น้ำเงิน"
    else: n = "ม่วง"
    return n, "#%02x%02x%02x" % (int(bgr[2]), int(bgr[1]), int(bgr[0]))

def _in_poly(pt, poly):
    x, y = pt; inside = False; n = len(poly); j = n-1
    for i in range(n):
        xi, yi = poly[i]; xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj-xi)*(y-yi)/(yj-yi+1e-9)+xi):
            inside = not inside
        j = i
    return inside

def _parse_start(start_time):
    """รับ 'YYYY/MM/DD HH:MM:SS' (จาก OCR) หรือ 'HH:MM:SS' -> datetime.
    ถ้ามีแต่เวลา ใช้ DEFAULT_DAY เป็นวันที่ฐาน"""
    s = (start_time or "").strip()
    for fmt in ("%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try: return datetime.strptime(s, fmt)
        except ValueError: pass
    m = re.search(r"(\d{1,2}):(\d{2}):(\d{2})", s)
    base = datetime.strptime(DEFAULT_DAY, "%Y-%m-%d")
    if m:
        h, mi, ss = (int(g) for g in m.groups())
        return base.replace(hour=h, minute=mi, second=ss)
    return base

def _store_load():
    if os.path.exists(STORE):
        return json.load(open(STORE, encoding="utf-8"))
    return {}

def _write_tracks_js():
    """รวม track ทุกกล้องใน store -> จับกลุ่มข้ามกล้อง (cosine) -> เขียน tracks_data.js"""
    store = _store_load()
    with db() as c:
        cams = {r["id"]: {"name": r["name"], "lat": r["lat"], "lon": r["lon"]}
                for r in c.execute("SELECT * FROM cameras")}
    keys = []
    for cid, tl in store.items():
        for t in tl:
            keys.append((cid, t["tid"]))
    # union-find mutual-best ข้ามกล้อง
    parent = {k: k for k in keys}
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    cids = list(store)
    for i in range(len(cids)):
        for jx in range(i+1, len(cids)):
            A = store[cids[i]]; B = store[cids[jx]]
            for ta in A:
                ea = np.array(ta["emb"]); best = None; bs = GROUP_TH
                for tb in B:
                    s = float(ea @ np.array(tb["emb"]))
                    if s > bs: bs = s; best = tb["tid"]
                if best is not None:
                    parent[find((cids[i], ta["tid"]))] = find((cids[jx], best))
    gid = {}
    def veh(k): r = find(k); return gid.setdefault(r, len(gid))
    out = []
    for cid, tl in store.items():
        cam = cams.get(cid, {"name": cid, "lat": 0, "lon": 0})
        for t in tl:
            out.append({"tid": f"{cid}-{t['tid']}", "veh": veh((cid, t["tid"])), "cam": cid,
                "camName": cam["name"], "lat": cam["lat"], "lon": cam["lon"],
                "time": t["time"], "date": t.get("date", DEFAULT_DAY), "img": t["img"],
                "color": t["color"], "colorHex": t["colorHex"], "type": t["type"],
                "typeTh": TH_CLS.get(t["type"], t["type"]), "emb": t["emb"]})
    out.sort(key=lambda t: (t["cam"], t["time"]))
    open(TRACKS_JS, "w", encoding="utf-8").write(
        "window.TDATA = " + json.dumps({"cameras": cams, "tracks": out}, ensure_ascii=False) + ";")
    return {"cameras": cams, "tracks": out}

# ---------------- OCR (lazy) ----------------
_READER = None
def reader():
    global _READER
    if _READER is None:
        import easyocr
        _READER = easyocr.Reader(["en"], gpu=(_device() != "cpu"), verbose=False)
    return _READER


class Api:
    # ----- video / frame -----
    def choose_video(self):
        win = webview.windows[0]
        res = win.create_file_dialog(webview.OPEN_DIALOG, allow_multiple=False,
            file_types=("Video (*.mp4;*.avi;*.mov;*.mkv)", "All files (*.*)"))
        if not res:
            return None
        path = res[0]
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return {"error": "เปิดวิดีโอไม่ได้ (ไฟล์เสีย/codec ไม่รองรับ)"}
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
        return {"path": path, "name": os.path.basename(path), "fps": round(fps, 2),
                "frames": n, "duration": _hms(n/fps) if fps else "—",
                "res": f"{w}x{h}", "size_gb": round(os.path.getsize(path)/1e9, 2)}

    def first_frame(self, path, at_frame=30, width=1100):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, at_frame)
        ok, fr = cap.read(); cap.release()
        if not ok:
            return None
        h, w = fr.shape[:2]; nh = int(h*width/w)
        fr = cv2.resize(fr, (width, nh))
        ok, buf = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 82])
        return "data:image/jpeg;base64," + base64.b64encode(buf).decode() if ok else None

    # ----- OCR timestamp (จริง) -----
    def read_timestamp(self, path, at_frame=30):
        """อ่าน timestamp จากแถบบนของเฟรมแรก -> {ok, datetime, box{x,y,w,h} normalized 0-1}"""
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return {"ok": False, "error": "เปิดวิดีโอไม่ได้"}
        cap.set(cv2.CAP_PROP_POS_FRAMES, at_frame)
        ok, fr = cap.read(); cap.release()
        if not ok:
            return {"ok": False, "error": "อ่านเฟรมไม่ได้"}
        H, W = fr.shape[:2]
        # crop เฉพาะมุมบนซ้าย (timestamp อยู่ตรงนี้) + ขยาย 2x ให้ OCR ชัด
        rw, rh = int(W*0.5), int(H*0.13)
        roi = cv2.resize(fr[:rh, :rw], (rw*2, rh*2))
        results = reader().readtext(roi, allowlist="0123456789/:.- ")
        joined = "".join(text.replace(" ", "") for _, text, _ in results)
        dt = None
        for mt in TS_RE.finditer(joined):           # หาคู่ที่ "สมเหตุผล" ตัวแรก
            y, mo, d, hh, mm, ss = (int(g) for g in mt.groups())
            if 1 <= mo <= 12 and 1 <= d <= 31 and hh < 24 and mm < 60 and ss < 60:
                dt = f"{y:04d}/{mo:02d}/{d:02d} {hh:02d}:{mm:02d}:{ss:02d}"; break
        if not dt:
            return {"ok": False, "error": "อ่าน timestamp ไม่ชัด — กรอกเอง", "raw": joined[:40]}
        # box รวม (map กลับสู่พิกัดเฟรมเต็ม, normalized)
        xs = [p[0] for b, _, _ in results for p in b]; ys = [p[1] for b, _, _ in results for p in b]
        bx = {"x": min(xs)/(rw*2)*0.5, "y": min(ys)/(rh*2)*0.13,
              "w": (max(xs)-min(xs))/(rw*2)*0.5, "h": (max(ys)-min(ys))/(rh*2)*0.13} if results else None
        return {"ok": True, "datetime": dt, "box": bx}

    # ----- cameras / paths CRUD -----
    def get_state(self):
        with db() as c:
            cams = [dict(r) for r in c.execute("SELECT * FROM cameras ORDER BY id")]
            paths = [dict(r) for r in c.execute("SELECT * FROM saved_paths ORDER BY created DESC")]
        for cam in cams:
            cam["roi"] = json.loads(cam["roi"] or "[]")
            cam["ts_box"] = json.loads(cam["ts_box"] or "null")
        for p in paths:
            p["sel"] = json.loads(p["sel"] or "{}")
        return {"cameras": cams, "saved_paths": paths}

    def add_camera(self, cam):
        cid = cam.get("id") or f"CAM-{int(time.time()*1000)%100000:05d}"
        with db() as c:
            c.execute("INSERT OR REPLACE INTO cameras VALUES(?,?,?,?,?,?,?,?,?)",
                (cid, cam.get("name",""), cam.get("lat"), cam.get("lon"),
                 cam.get("video_path",""), cam.get("start_time",""),
                 json.dumps(cam.get("roi",[]), ensure_ascii=False),
                 json.dumps(cam.get("ts_box"), ensure_ascii=False),
                 time.strftime("%Y-%m-%d %H:%M")))
        return cid

    def update_camera(self, cid, fields):
        allowed = {"name","lat","lon","video_path","start_time"}
        sets = {k: v for k, v in fields.items() if k in allowed}
        if "roi" in fields: sets["roi"] = json.dumps(fields["roi"], ensure_ascii=False)
        if "ts_box" in fields: sets["ts_box"] = json.dumps(fields["ts_box"], ensure_ascii=False)
        if not sets: return False
        with db() as c:
            c.execute(f"UPDATE cameras SET {','.join(k+'=?' for k in sets)} WHERE id=?",
                      [*sets.values(), cid])
        return True

    def delete_camera(self, cid):
        with db() as c:
            c.execute("DELETE FROM cameras WHERE id=?", (cid,))
        # ลบ track ของกล้องนี้ออกจาก store + ลบรูป crop + เขียน tracks_data.js ใหม่
        store = _store_load()
        for t in store.pop(cid, []):
            img = os.path.join(APPDIR, (t.get("img") or "").replace("/", os.sep))
            try:
                if os.path.isfile(img): os.remove(img)
            except OSError: pass
        json.dump(store, open(STORE, "w", encoding="utf-8"))
        data = _write_tracks_js()
        return {"ok": True, "data": data}

    def save_path(self, p):
        pid = p.get("id") or f"P-{int(time.time()*1000)%1000000:06d}"
        with db() as c:
            c.execute("INSERT OR REPLACE INTO saved_paths VALUES(?,?,?,?,?)",
                (pid, p.get("name",""), p.get("note",""),
                 time.strftime("%d/%m/%Y %H:%M"), json.dumps(p.get("sel",{}), ensure_ascii=False)))
        return pid

    def delete_path(self, pid):
        with db() as c:
            c.execute("DELETE FROM saved_paths WHERE id=?", (pid,))
        return True

    # ----- pipeline: ประมวลผล background + progress -----
    def start_processing(self, cam_id, minutes=PROC_MINUTES):
        if PROGRESS.get(cam_id, {}).get("running"):
            return {"ok": False, "error": "กำลังประมวลผลอยู่"}
        if any(p.get("running") for p in PROGRESS.values()):
            return {"ok": False, "error": "มีกล้องอื่นกำลังประมวลผล รอให้เสร็จก่อน"}
        PROGRESS[cam_id] = {"pct": 0, "stage": "เริ่ม", "done": False, "running": True}
        threading.Thread(target=_process_worker, args=(cam_id, minutes), daemon=True).start()
        return {"ok": True, "started": True}

    def proc_status(self, cam_id):
        return PROGRESS.get(cam_id) or {"pct": 0, "done": False, "stage": "—"}

    def cancel_processing(self, cam_id):
        p = PROGRESS.get(cam_id)
        if p and p.get("running"):
            p["cancel"] = True
            return {"ok": True}
        return {"ok": False, "error": "ไม่มีงานที่กำลังประมวลผล"}


def _process_worker(cam_id, minutes):
    """รัน detect+track+reid บน thread แยก พร้อมอัปเดต PROGRESS[cam_id]['pct']"""
    P = PROGRESS[cam_id]
    try:
        with db() as c:
            row = c.execute("SELECT * FROM cameras WHERE id=?", (cam_id,)).fetchone()
        if not row: raise RuntimeError("ไม่พบกล้อง")
        cam = dict(row); path = cam.get("video_path", "")
        if not path or not os.path.exists(path): raise RuntimeError("ไม่พบไฟล์วิดีโอ")
        roi = json.loads(cam.get("roi") or "[]"); start_dt = _parse_start(cam.get("start_time"))
        dev = _device()
        P["stage"] = "โหลดโมเดล"; model = _yolo(); ext = _ext()
        cap = cv2.VideoCapture(path)
        if not cap.isOpened(): raise RuntimeError("เปิดวิดีโอไม่ได้")
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 24
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1); Hh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1)
        step = max(1, round(src_fps/12)); end_s = minutes*60
        model.predictor = None
        from collections import Counter
        tracks = {}; nfr = 0; P["stage"] = "ตรวจจับ+ติดตาม"
        while True:
            if P.get("cancel"):
                cap.release()
                P.update({"done": True, "running": False, "cancelled": True, "stage": "ยกเลิกแล้ว"})
                return
            if not cap.grab(): break
            pos = cap.get(cv2.CAP_PROP_POS_MSEC)/1000
            if pos >= end_s: break
            P["pct"] = int(min(pos/end_s, 1.0)*88)        # 0-88% = detect/track
            if nfr % step != 0: nfr += 1; continue
            ok, fr = cap.retrieve()
            if not ok: break
            nfr += 1
            r = model.track(fr, persist=True, tracker="bytetrack.yaml",
                            classes=[2,3,5,7], imgsz=640, conf=0.3, verbose=False, device=dev)[0]
            if r.boxes is None or r.boxes.id is None: continue
            xy = r.boxes.xyxy.cpu().numpy(); ids = r.boxes.id.cpu().numpy().astype(int)
            cls = r.boxes.cls.cpu().numpy().astype(int); cf = r.boxes.conf.cpu().numpy()
            for (x1,y1,x2,y2), tid, cc, cv_ in zip(xy, ids, cls, cf):
                x1,y1,x2,y2 = max(0,int(x1)),max(0,int(y1)),int(x2),int(y2)
                if x2-x1 < 12 or y2-y1 < 12: continue
                if roi and len(roi) >= 3:
                    cx, cy = (x1+x2)/2/W, (y1+y2)/2/Hh
                    if not _in_poly((cx, cy), roi): continue
                T = tracks.get(tid) or tracks.setdefault(tid, {"n":0,"conf":[],"cls":Counter(),"first":pos,"crops":[]})
                T["n"] += 1; T["conf"].append(float(cv_)); T["cls"][COCO.get(int(cc),"car")] += 1
                T["crops"].append(((x2-x1)*(y2-y1), fr[y1:y2, x1:x2].copy()))
                T["crops"].sort(key=lambda a: -a[0]); T["crops"] = T["crops"][:5]
        cap.release()
        P["stage"] = "Re-ID"; P["pct"] = 90
        items = [(tid, T) for tid, T in tracks.items() if T["n"] >= 8 and float(np.mean(T["conf"])) >= 0.40]
        out = []
        for i, (tid, T) in enumerate(items):
            feats = [ext.embed_bgr(c) for _, c in T["crops"]]
            emb = np.mean(feats, axis=0); emb /= (np.linalg.norm(emb)+1e-9)
            best = max(T["crops"], key=lambda a: a[0])[1]
            imgname = f"t_{cam_id}_{tid}.jpg"; cv2.imwrite(os.path.join(IMGDIR, imgname), best)
            cname, chex = _color(best); kls = T["cls"].most_common(1)[0][0]
            real = start_dt + timedelta(seconds=T["first"])
            out.append({"tid": str(tid), "time": real.strftime("%H:%M:%S"), "date": real.strftime("%Y-%m-%d"),
                "img": f"img/{imgname}", "color": cname, "colorHex": chex, "type": kls,
                "emb": [round(float(x), 4) for x in emb]})
            P["pct"] = 90 + int((i+1)/max(len(items),1)*8)   # 90-98%
        P["stage"] = "บันทึก"; P["pct"] = 99
        store = _store_load(); store[cam_id] = out
        json.dump(store, open(STORE, "w", encoding="utf-8"))
        data = _write_tracks_js()
        P.update({"pct": 100, "done": True, "running": False, "stage": "เสร็จ",
                  "n": len(out), "total": len(data["tracks"]), "data": data})
    except Exception as e:
        import traceback; traceback.print_exc()
        P.update({"done": True, "running": False, "error": str(e), "stage": "ผิดพลาด"})


def main():
    init_db()
    webview.create_window("TaThip — ค้นหา & ติดตามยานพาหนะข้ามกล้อง",
        INDEX, js_api=Api(), width=1340, height=900, min_size=(1024, 700))
    webview.start()


if __name__ == "__main__":
    main()
