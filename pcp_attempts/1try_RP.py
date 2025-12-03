#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import cv2
import numpy as np
import serial
import serial.tools.list_ports
import time
import sys
import os
import signal

def find_arduino_port():
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        name = (p.description or "") + " " + (p.manufacturer or "") + " " + (p.device or "")
        name_l = name.lower()
        if any(s in name_l for s in ["arduino", "wch", "acm", "usb serial"]):
            return p.device
    # типичные дефолты
    for dev in ["/dev/ttyACM0", "/dev/ttyUSB0", "/dev/ttyACM1", "/dev/ttyUSB1"]:
        if os.path.exists(dev):
            return dev
    return None

def open_serial(port, baud, timeout=0.01):
    try:
        ser = serial.Serial(port, baudrate=baud, timeout=timeout)
        time.sleep(0.2)
        return ser
    except Exception as e:
        print(f"[UART] Не удалось открыть порт {port}: {e}", file=sys.stderr)
        return None

def open_c270_linux(preferred_index=None, width=1280, height=720, fps=30):
    backend = cv2.CAP_V4L2
    tried = []
    candidates = []
    if isinstance(preferred_index, int):
        candidates.append(preferred_index)
    candidates += [0, 1, 2, 3, 4, 5]
    used = set()
    candidates = [c for c in candidates if not (c in used or used.add(c))]

    for idx in candidates:
        cap = cv2.VideoCapture(idx, backend)
        if not cap.isOpened():
            tried.append(f"{idx}")
            continue
        # Настроить MJPG + режим
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        except Exception:
            pass
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        time.sleep(0.15)
        ok, _ = cap.read()
        if ok:
            return cap, f"/dev/video{idx}"
        cap.release()
        tried.append(f"{idx}")
    return None, f"не найдено (пробовали индексы: {', '.join(tried)})"

def build_masks(hsv, params):
    # Красный: два диапазона H
    h1l, h1u = params["red_h1"]
    h2l, h2u = params["red_h2"]
    sl, su   = params["red_s"]
    vl, vu   = params["red_v"]

    lower_red1 = np.array([h1l, sl, vl], dtype=np.uint8)
    upper_red1 = np.array([h1u, su, vu], dtype=np.uint8)
    lower_red2 = np.array([h2l, sl, vl], dtype=np.uint8)
    upper_red2 = np.array([h2u, su, vu], dtype=np.uint8)

    mask_r1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_r2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask_r1, mask_r2)

    # Белый: низкая S, высокая V
    ws_max = params["white_s_max"]
    wv_min = params["white_v_min"]
    lower_white = np.array([0, 0, wv_min], dtype=np.uint8)
    upper_white = np.array([179, ws_max, 255], dtype=np.uint8)
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # Морфология
    k = params["k"]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    if params["close_iters"] > 0:
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=params["close_iters"])
    if params["open_iters"] > 0:
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN,  kernel, iterations=params["open_iters"])
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel, iterations=params["open_iters"])
    return red_mask, white_mask

def detect_target(frame, params):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red_mask, white_mask = build_masks(hsv, params)

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cx, cy, wmax, hmax = None, None, 0, 0
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area >= params["min_area"]:
            rect = cv2.minAreaRect(cnt)  # ((cx, cy), (w, h), angle)
            box = cv2.boxPoints(rect).astype(np.int32)
            xs = box[:, 0]
            ys = box[:, 1]
            x_min, x_max = int(xs.min()), int(xs.max())
            y_min, y_max = int(ys.min()), int(ys.max())

            cx = int((x_min + x_max) / 2)
            cy = int((y_min + y_max) / 2)
            wmax = int(x_max - x_min)
            hmax = int(y_max - y_min)

            # Проверка доли белого внутри красного контура
            validate = True
            white_ratio_req = float(params["white_ratio"])
            if white_ratio_req > 0:
                poly_mask = np.zeros_like(white_mask)
                cv2.fillPoly(poly_mask, [box], 255)
                inside_area = cv2.countNonZero(poly_mask)
                if inside_area > 0:
                    white_inside = cv2.bitwise_and(white_mask, poly_mask)
                    white_count = cv2.countNonZero(white_inside)
                    ratio = (white_count / inside_area) * 100.0
                    validate = ratio >= white_ratio_req
                else:
                    validate = False

            if not validate:
                cx, cy, wmax, hmax = None, None, 0, 0

    return cx, cy, wmax, hmax

def send_uart(ser, cx, cy, w, h):
    if ser is None:
        return
    if cx is None or cy is None:
        line = "null,null,null,null\n"
    else:
        line = f"{int(cx)},{int(cy)},{int(w)},{int(h)}\n"
    try:
        ser.write(line.encode('ascii'))
    except Exception as e:
        print(f"[UART] Ошибка записи: {e}", file=sys.stderr)

def clamp_pair(a, b, mn, mx):
    a, b = int(a), int(b)
    a, b = min(a, b), max(a, b)
    a = max(mn, min(mx, a))
    b = max(mn, min(mx, b))
    return a, b

def make_params_from_args(args):
    h1l, h1u = clamp_pair(args.red_h1[0], args.red_h1[1], 0, 179)
    h2l, h2u = clamp_pair(args.red_h2[0], args.red_h2[1], 0, 179)
    sl,  su  = clamp_pair(args.red_s[0],  args.red_s[1],  0, 255)
    vl,  vu  = clamp_pair(args.red_v[0],  args.red_v[1],  0, 255)
    k = int(args.kernel)
    if k < 1: k = 1
    if k % 2 == 0: k += 1
    return {
        "red_h1": (h1l, h1u),
        "red_h2": (h2l, h2u),
        "red_s":  (sl, su),
        "red_v":  (vl, vu),
        "white_s_max": int(args.white_s_max),
        "white_v_min": int(args.white_v_min),
        "k": k,
        "open_iters": int(args.open_iters),
        "close_iters": int(args.close_iters),
        "min_area": max(0, int(args.min_area)),
        "white_ratio": float(args.white_ratio),
    }

def parse_args():
    ap = argparse.ArgumentParser(description="Headless детектор мишени (Raspberry Pi + Logitech C270) -> UART")
    ap.add_argument("--cam-index", type=int, default=None, help="Индекс камеры (/dev/videoX). Если не задан, пробуем 0..5")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)

    ap.add_argument("--serial", type=str, default="auto", help="Путь к UART (например, /dev/ttyACM0). 'auto' — автопоиск")
    ap.add_argument("--baud", type=int, default=115200)

    # HSV пороги
    ap.add_argument("--red-h1", nargs=2, type=int, default=[0, 12],   metavar=("L","U"))
    ap.add_argument("--red-h2", nargs=2, type=int, default=[170, 179],metavar=("L","U"))
    ap.add_argument("--red-s",  nargs=2, type=int, default=[80, 255], metavar=("L","U"))
    ap.add_argument("--red-v",  nargs=2, type=int, default=[50, 255], metavar=("L","U"))

    ap.add_argument("--white-s-max", type=int, default=60)
    ap.add_argument("--white-v-min", type=int, default=200)

    ap.add_argument("--kernel",     type=int, default=5, help="Размер ядра морфологии (нечётный)")
    ap.add_argument("--open-iters", type=int, default=1)
    ap.add_argument("--close-iters",type=int, default=2)
    ap.add_argument("--min-area",   type=int, default=1500)
    ap.add_argument("--white-ratio",type=float, default=5.0, help="Мин. %% белого внутри красного")

    ap.add_argument("--send-every", type=int, default=1, help="Отправлять каждую N-ю рамку (нагрузка на UART)")
    ap.add_argument("--log-interval", type=float, default=2.0, help="Лог в консоль каждые N секунд")
    return ap.parse_args()

def main():
    args = parse_args()

    # UART
    port = find_arduino_port() if args.serial == "auto" else args.serial
    if not port:
        print("[UART] Порт не найден (подключён ли Arduino?). Можно указать вручную: --serial /dev/ttyACM0", file=sys.stderr)
    ser = open_serial(port, args.baud) if port else None
    if ser:
        print(f"[UART] Открыт {ser.port} @ {args.baud}")

    # Камера
    cap, desc = open_c270_linux(preferred_index=args.cam_index, width=args.width, height=args.height, fps=args.fps)
    if cap is None:
        print(f"[Camera] Камера не открыта: {desc}", file=sys.stderr)
        sys.exit(1)
    print(f"[Camera] Открыта: {desc} {args.width}x{args.height}@{args.fps} (MJPG)")

    params = make_params_from_args(args)

    # Грациозное завершение по Ctrl+C
    stop = False
    def _sig_handler(sig, frm):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    last_log = 0.0
    frame_id = 0
    t0 = time.time()

    while not stop:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("[Camera] Кадр не получен, пробуем дальше...", file=sys.stderr)
            time.sleep(0.05)
            continue

        cx, cy, wmax, hmax = detect_target(frame, params)

        frame_id += 1
        if frame_id % max(1, args.send_every) == 0:
            send_uart(ser, cx, cy, wmax, hmax)

        now = time.time()
        if (now - last_log) >= args.log_interval:
            last_log = now
            fps_est = frame_id / max(1e-6, (now - t0))
            status = f"{cx},{cy},{wmax},{hmax}" if cx is not None else "null,null,null,null"
            print(f"[RUN] fps={fps_est:.1f}  last={status}")

    cap.release()
    if ser:
        ser.close()
    print("[OK] Завершено.")

if __name__ == "__main__":
    main()