#!/usr/bin/env python3
import cv2
import time
import serial
import serial.tools.list_ports
import numpy as np
import sys
import os

# --- Камера и детектор ---
CAM_INDEX = 0
WIDTH, HEIGHT, FPS = 1280, 720, 30

# Пороговые значения для красного и белого (в HSV)
RED_H1 = (0, 12)
RED_H2 = (170, 179)
RED_S  = (80, 255)
RED_V  = (50, 255)
WHITE_S_MAX = 60
WHITE_V_MIN = 200

MORPH_K = 5
OPEN_ITERS, CLOSE_ITERS = 1, 2
MIN_AREA = 1500
WHITE_RATIO = 5.0  # % белого внутри красного

# --- Геометрия камеры и наведение ---
H_FOV_DEG = 60.0   # оценочно для C270
V_FOV_DEG = 34.0
TRACK_K = 0.8       # скорость «догонки» (0.3..1.2)
MAX_ANGLE = 90.0    # софт-лимиты на Arduino тоже стоят

# Центр «попадание»
CENTER_TOL_PAN_DEG  = 1.5
CENTER_TOL_TILT_DEG = 1.5
HIT_HOLD_SEC        = 0.25
LOG_FILE            = "/home/pi/center_hits.log"

# Возврат к нулю, если цель потеряна
LOST_FRAMES_TO_RECENTER = 45
LOST_RET_ALPHA = 0.98

# --- Связь с Arduino ---
SER_BAUD = 115200
SEND_HZ = 20
SER_TIMEOUT = 0.02

def find_arduino_port():
    # Пытаемся найти /dev/ttyACM* или by-id с “Arduino”
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if ("Arduino" in (p.description or "")) or ("CDC" in (p.description or "")):
            return p.device
    # fallback
    for cand in ("/dev/ttyACM0", "/dev/ttyUSB0"):
        if os.path.exists(cand):
            return cand
    return None

def open_cam(dev_index=CAM_INDEX):
    backend = getattr(cv2, "CAP_V4L2", 0)
    cap = cv2.VideoCapture(dev_index, backend)
    if not cap.isOpened():
        return None
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    except Exception:
        pass
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    time.sleep(0.2)
    ok, _ = cap.read()
    if not ok:
        cap.release()
        return None
    return cap

def detect_target(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    (h1l,h1u),(h2l,h2u) = RED_H1, RED_H2
    (sl,su),(vl,vu) = RED_S, RED_V

    red1 = cv2.inRange(hsv, np.array([h1l,sl,vl],np.uint8), np.array([h1u,su,vu],np.uint8))
    red2 = cv2.inRange(hsv, np.array([h2l,sl,vl],np.uint8), np.array([h2u,su,vu],np.uint8))
    red_mask = cv2.bitwise_or(red1, red2)

    white_mask = cv2.inRange(
        hsv,
        np.array([0, 0, WHITE_V_MIN], np.uint8),
        np.array([179, WHITE_S_MAX, 255], np.uint8)
    )

    k = MORPH_K if MORPH_K % 2 == 1 else MORPH_K + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    if CLOSE_ITERS:
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=CLOSE_ITERS)
    if OPEN_ITERS:
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN,  kernel, iterations=OPEN_ITERS)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel, iterations=OPEN_ITERS)

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cx, cy, wmax, hmax = None, None, 0, 0
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area >= MIN_AREA:
            rect = cv2.minAreaRect(cnt)
            pts = cv2.boxPoints(rect).astype(int)
            xs, ys = pts[:,0], pts[:,1]
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            cx, cy = int((x_min + x_max) / 2), int((y_min + y_max) / 2)
            wmax, hmax = int(x_max - x_min), int(y_max - y_min)

            if WHITE_RATIO > 0:
                poly_mask = np.zeros_like(white_mask)
                cv2.fillPoly(poly_mask, [pts], 255)
                inside_area = cv2.countNonZero(poly_mask)
                if inside_area > 0:
                    white_inside = cv2.countNonZero(cv2.bitwise_and(white_mask, poly_mask))
                    ratio = 100.0 * white_inside / inside_area
                    if ratio < WHITE_RATIO:
                        cx, cy, wmax, hmax = None, None, 0, 0
                else:
                    cx, cy, wmax, hmax = None, None, 0, 0
    return cx, cy, wmax, hmax

class ArduinoGimbal:
    def __init__(self, baud=SER_BAUD, timeout=SER_TIMEOUT, send_hz=SEND_HZ):
        self.last_send = 0.0
        self.send_dt = 1.0 / max(1, send_hz)
        port = find_arduino_port()
        if not port:
            print("[SER] Arduino port not found")
            self.ser = None
            return
        try:
            self.ser = serial.Serial(port, baudrate=baud, timeout=timeout)
            time.sleep(2.0)  # reset on DTR
            self.ser.reset_input_buffer()
            self.ser.write(b"Z\n")  # принять текущее как ноль
            print(f"[SER] connected: {port} @ {baud}")
        except Exception as e:
            print(f"[SER] ERROR: {e}")
            self.ser = None

    def send_angles(self, pan_deg, tilt_deg):
        if not self.ser: return
        now = time.time()
        if now - self.last_send < self.send_dt: return
        try:
            self.ser.write(f"A {pan_deg:.2f} {tilt_deg:.2f}\n".encode("ascii"))
            self.last_send = now
        except Exception as e:
            print(f"[SER] write error: {e}")

def clamp(v, lo, hi): return max(lo, min(hi, v))

def ensure_logfile():
    try:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        if not os.path.exists(LOG_FILE):
            with open(LOG_FILE, "w") as f:
                f.write("# ts,cx,cy,w,h,err_pan,err_tilt,pan_set,tilt_set,event,dwell\n")
    except Exception:
        pass

def append_log(ts, cx, cy, w, h, e_pan, e_tilt, pan_set, tilt_set, event, dwell=0.0):
    line = f"{ts:.3f},{cx},{cy},{w},{h},{e_pan:.3f},{e_tilt:.3f},{pan_set:.2f},{tilt_set:.2f},{event},{dwell:.3f}\n"
    try:
        with open(LOG_FILE, "a") as f:
            f.write(line)
    except Exception:
        pass

def main():
    cap = open_cam()
    if cap is None:
        print("Камера не открылась. Проверьте /dev/video0 и v4l2-ctl --list-devices")
        sys.exit(1)

    ard = ArduinoGimbal()
    ensure_logfile()

    pan_set = 0.0
    tilt_set = 0.0
    lost_frames = 0

    in_center = False
    pre_enter_time = None
    enter_time = None
    hits = 0

    t0 = time.time()
    fps_cnt = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Кадр не получен.")
                break

            cx, cy, wmax, hmax = detect_target(frame)
            h, w = frame.shape[:2]

            if cx is not None:
                lost_frames = 0
                err_pan_deg  = ((cx - (w/2)) / (w/2)) * (H_FOV_DEG / 2.0)
                err_tilt_deg = -((cy - (h/2)) / (h/2)) * (V_FOV_DEG / 2.0)

                pan_set  = clamp(pan_set  + TRACK_K * err_pan_deg,  -MAX_ANGLE, MAX_ANGLE)
                tilt_set = clamp(tilt_set + TRACK_K * err_tilt_deg, -MAX_ANGLE, MAX_ANGLE)

                center_now = (abs(err_pan_deg) <= CENTER_TOL_PAN_DEG) and (abs(err_tilt_deg) <= CENTER_TOL_TILT_DEG)
                now = time.time()
                if center_now and not in_center:
                    if pre_enter_time is None:
                        pre_enter_time = now
                    elif now - pre_enter_time >= HIT_HOLD_SEC:
                        in_center = True
                        enter_time = now
                        hits += 1
                        print(f"[HIT] #{hits} err=({err_pan_deg:.2f}°, {err_tilt_deg:.2f}°) set=({pan_set:.1f}°, {tilt_set:.1f}°) size={wmax}x{hmax}")
                        append_log(now, cx, cy, wmax, hmax, err_pan_deg, err_tilt_deg, pan_set, tilt_set, "HIT", 0.0)
                elif not center_now:
                    pre_enter_time = None
                    if in_center:
                        now = time.time()
                        dwell = now - (enter_time if enter_time else now)
                        print(f"[LEAVE] after {dwell:.2f}s")
                        append_log(now, cx if cx is not None else -1, cy if cy is not None else -1,
                                   wmax, hmax, err_pan_deg, err_tilt_deg, pan_set, tilt_set, "LEAVE", dwell)
                        in_center = False
                        enter_time = None
            else:
                lost_frames += 1
                if lost_frames > LOST_FRAMES_TO_RECENTER:
                    pan_set  *= LOST_RET_ALPHA
                    tilt_set *= LOST_RET_ALPHA
                    if abs(pan_set) < 0.2: pan_set = 0.0
                    if abs(tilt_set) < 0.2: tilt_set = 0.0
                pre_enter_time = None
                if in_center:
                    now = time.time()
                    dwell = now - (enter_time if enter_time else now)
                    print(f"[LEAVE] target lost after {dwell:.2f}s")
                    append_log(now, -1, -1, 0, 0, 0.0, 0.0, pan_set, tilt_set, "LEAVE", dwell)
                    in_center = False
                    enter_time = None

            # Отправка на Arduino
            ard.send_angles(pan_set, tilt_set)

            # Статистика раз в ~1с
            fps_cnt += 1
            now = time.time()
            if now - t0 >= 1.0:
                fps = fps_cnt / (now - t0)
                t0 = now
                fps_cnt = 0
                ctr = "YES" if in_center else "no"
                print(f"[STAT] center={ctr} set=({pan_set:.1f}°, {tilt_set:.1f}°) FPS={fps:.1f}")

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        print("Выход.")

if __name__ == "__main__":
    main()