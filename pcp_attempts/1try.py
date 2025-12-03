#!/usr/bin/env python3
import cv2
import numpy as np
import time
import signal
import sys

# Пытаемся подключить pigpio (управление сервоприводом)
try:
    import pigpio
except ImportError:
    pigpio = None
    print("Внимание: модуль pigpio не найден. Установите: sudo apt install python3-pigpio pigpio")

# ------------- ПАРАМЕТРЫ СЕРВО -------------
SERVO_GPIO = 18          # GPIO для сигнала сервопривода (физ. пин 12)
SERVO_MIN_US = 500       # минимальный импульс, мкс (калибруется под вашу серву)
SERVO_MAX_US = 2500      # максимальный импульс, мкс
SERVO_MID_US = 1500      # центр
SERVO_INVERT = False     # инвертировать направление (True/False)
SERVO_ALPHA = 0.2        # сглаживание (0..1), больше -> быстрее реагирует, меньше -> плавнее
RET2CENTER_AFTER_LOST_FRAMES = 30  # через сколько кадров без цели возвращаться к центру

# Сколько использовать диапазона от центра (в мкс).
# Например, если min=500, max=2500, половина диапазона = 1000 мкс. Чуть уменьшим, чтобы не биться в упоры:
SERVO_HALF_SPAN = int((SERVO_MAX_US - SERVO_MIN_US) * 0.45)  # ~90% от половины

# ------------- ВСПОМОГАТЕЛЬНОЕ -------------
def open_c270_v4l2(preferred_index=None, width=1280, height=720, fps=30):
    """Открыть Logitech C270 в Linux через V4L2."""
    indices = [preferred_index] if isinstance(preferred_index, int) else [0, 1, 2, 3, 4]
    for idx in indices:
        try:
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2) if hasattr(cv2, "CAP_V4L2") else cv2.VideoCapture(idx)
            if not cap.isOpened():
                continue
            # Предпочтительно MJPG на C270
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            time.sleep(0.2)
            ok, _ = cap.read()
            if ok:
                return cap, f"/dev/video{idx}"
            cap.release()
        except Exception:
            pass
    return None, "not found"

def create_trackbars(win="Параметры"):
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 520, 420)

    # Красный (два диапазона H), общие S/V
    cv2.createTrackbar("Red H1 L", win, 0,   179, lambda x: None)
    cv2.createTrackbar("Red H1 U", win, 12,  179, lambda x: None)
    cv2.createTrackbar("Red H2 L", win, 170, 179, lambda x: None)
    cv2.createTrackbar("Red H2 U", win, 179, 179, lambda x: None)
    cv2.createTrackbar("Red S L",  win, 80,  255, lambda x: None)
    cv2.createTrackbar("Red S U",  win, 255, 255, lambda x: None)
    cv2.createTrackbar("Red V L",  win, 50,  255, lambda x: None)
    cv2.createTrackbar("Red V U",  win, 255, 255, lambda x: None)

    # Белый
    cv2.createTrackbar("White S max", win, 60, 255, lambda x: None)
    cv2.createTrackbar("White V min", win, 200,255, lambda x: None)

    # Морфология и фильтры
    cv2.createTrackbar("Morph k",      win, 5, 21, lambda x: None)   # размер ядра (нечётный)
    cv2.createTrackbar("Open iters",   win, 1, 5,  lambda x: None)
    cv2.createTrackbar("Close iters",  win, 2, 5,  lambda x: None)
    cv2.createTrackbar("Min area",     win, 1000, 50000, lambda x: None)
    cv2.createTrackbar("White ratio %",win, 5, 100, lambda x: None)  # % белого внутри красного

    return win

def get_params(win):
    h1l = cv2.getTrackbarPos("Red H1 L", win)
    h1u = cv2.getTrackbarPos("Red H1 U", win)
    h2l = cv2.getTrackbarPos("Red H2 L", win)
    h2u = cv2.getTrackbarPos("Red H2 U", win)
    sl = cv2.getTrackbarPos("Red S L", win)
    su = cv2.getTrackbarPos("Red S U", win)
    vl = cv2.getTrackbarPos("Red V L", win)
    vu = cv2.getTrackbarPos("Red V U", win)

    ws_max = cv2.getTrackbarPos("White S max", win)
    wv_min = cv2.getTrackbarPos("White V min", win)

    k     = cv2.getTrackbarPos("Morph k",     win)
    iopen = cv2.getTrackbarPos("Open iters",  win)
    iclose= cv2.getTrackbarPos("Close iters", win)
    min_area = cv2.getTrackbarPos("Min area", win)
    wratio = cv2.getTrackbarPos("White ratio %", win)

    # Упорядочим H
    h1l, h1u = min(h1l, h1u), max(h1l, h1u)
    h2l, h2u = min(h2l, h2u), max(h2l, h2u)

    if k < 1: k = 1
    if k % 2 == 0: k += 1

    return {
        "red_h1": (h1l, h1u),
        "red_h2": (h2l, h2u),
        "red_s":  (sl, su),
        "red_v":  (vl, vu),
        "white_s_max": ws_max,
        "white_v_min": wv_min,
        "k": k,
        "open_iters": iopen,
        "close_iters": iclose,
        "min_area": max(0, min_area),
        "white_ratio": max(0.0, min(100.0, float(wratio))),
    }

class ServoController:
    def __init__(self, gpio=SERVO_GPIO):
        self.pi = None
        self.gpio = gpio
        self.enabled = False
        self.current_us = SERVO_MID_US
        self.lost_frames = 0

        if pigpio is None:
            print("pigpio недоступен — управление сервоприводом отключено.")
            return

        self.pi = pigpio.pi()
        if not self.pi.connected:
            print("Не удалось подключиться к pigpio daemon. Запустите: sudo systemctl start pigpio")
            return

        self.pi.set_servo_pulsewidth(self.gpio, SERVO_MID_US)
        self.enabled = True
        print(f"Серво инициализировано на GPIO{self.gpio}, в центре ({SERVO_MID_US} мкс).")

    def shutdown(self):
        if self.pi and self.enabled:
            self.pi.set_servo_pulsewidth(self.gpio, 0)  # отпустить серво
            self.pi.stop()
            self.enabled = False

    def update(self, cx, frame_w):
        """Обновить позицию сервопривода по центру цели cx (или None, если цели нет)."""
        if not self.enabled:
            return self.current_us

        if cx is None:
            # цели нет
            self.lost_frames += 1
            if self.lost_frames > RET2CENTER_AFTER_LOST_FRAMES:
                # мягко возвращаемся к центру
                target = SERVO_MID_US
                self.current_us = int((1 - SERVO_ALPHA) * self.current_us + SERVO_ALPHA * target)
                self.pi.set_servo_pulsewidth(self.gpio, self.current_us)
            return self.current_us

        # цель есть
        self.lost_frames = 0
        dx = (cx - (frame_w / 2)) / (frame_w / 2)  # -1..+1
        if SERVO_INVERT:
            dx = -dx

        target = SERVO_MID_US + int(dx * SERVO_HALF_SPAN)
        target = max(SERVO_MIN_US, min(SERVO_MAX_US, target))

        # сглаживание
        self.current_us = int((1 - SERVO_ALPHA) * self.current_us + SERVO_ALPHA * target)
        self.pi.set_servo_pulsewidth(self.gpio, self.current_us)
        return self.current_us

def main():
    # Камера
    cap, desc = open_c270_v4l2(preferred_index=None)
    if cap is None:
        print("Не удалось открыть Logitech C270. Проверь: v4l2-ctl --list-devices")
        return

    ctrl_win = create_trackbars()

    main_win = f"Camera (C270) [{desc}]"
    cv2.namedWindow(main_win, cv2.WINDOW_NORMAL)
    cv2.namedWindow("Red mask", cv2.WINDOW_NORMAL)
    cv2.namedWindow("White mask", cv2.WINDOW_NORMAL)

    # Серво
    servo = ServoController(SERVO_GPIO)

    # Корректное завершение по Ctrl+C
    def _sigint(_a, _b):
        cap.release()
        if servo:
            servo.shutdown()
        cv2.destroyAllWindows()
        sys.exit(0)
    signal.signal(signal.SIGINT, _sigint)

    fps_t0 = time.time()
    fps_cnt = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Кадр не получен.")
            break

        params = get_params(ctrl_win)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Красная маска (два диапазона H)
        (h1l, h1u) = params["red_h1"]
        (h2l, h2u) = params["red_h2"]
        (sl, su)   = params["red_s"]
        (vl, vu)   = params["red_v"]

        lower_red1 = np.array([h1l, sl, vl], dtype=np.uint8)
        upper_red1 = np.array([h1u, su, vu], dtype=np.uint8)
        lower_red2 = np.array([h2l, sl, vl], dtype=np.uint8)
        upper_red2 = np.array([h2u, su, vu], dtype=np.uint8)

        mask_r1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_r2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask_r1, mask_r2)

        # Белая маска
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

        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cx, cy = None, None
        wmax, hmax = 0, 0

        if contours:
            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)

            if area >= params["min_area"]:
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect).astype(int)
                xs = box[:, 0]
                ys = box[:, 1]
                x_min, x_max = xs.min(), xs.max()
                y_min, y_max = ys.min(), ys.max()

                cx = int((x_min + x_max) / 2)
                cy = int((y_min + y_max) / 2)

                wmax = int(x_max - x_min)
                hmax = int(y_max - y_min)

                # Проверка доли белого внутри красного
                valid = True
                white_ratio_req = params["white_ratio"]
                if white_ratio_req > 0:
                    poly_mask = np.zeros_like(white_mask)
                    cv2.fillPoly(poly_mask, [box], 255)
                    inside_area = cv2.countNonZero(poly_mask)
                    if inside_area > 0:
                        white_inside = cv2.bitwise_and(white_mask, poly_mask)
                        white_count = cv2.countNonZero(white_inside)
                        ratio = (white_count / inside_area) * 100.0
                        valid = ratio >= white_ratio_req
                    else:
                        valid = False

                if valid:
                    cv2.polylines(frame, [box], True, (0, 255, 255), 2)
                    cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)
                else:
                    cx, cy = None, None
                    wmax, hmax = 0, 0

        # Обновление сервопривода
        h_img, w_img = frame.shape[:2]
        servo_us = None
        if servo:
            servo_us = servo.update(cx, w_img)

        # ФПС
        fps_cnt += 1
        if fps_cnt >= 10:
            dt = time.time() - fps_t0
            fps = fps_cnt / dt if dt > 0 else 0.0
            fps_t0 = time.time()
            fps_cnt = 0
        else:
            fps = None

        # Подписи
        text = f"Center: { (cx, cy) if cx is not None else '—' } | W:{wmax}px H:{hmax}px"
        if servo_us is not None:
            text += f" | Servo: {servo_us}us"
        if fps is not None:
            text += f" | FPS: {fps:.1f}"

        cv2.rectangle(frame, (0, h_img - 30), (w_img, h_img), (0, 0, 0), thickness=cv2.FILLED)
        cv2.putText(frame, text, (10, h_img - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0) if cx is not None else (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow(main_win, frame)
        cv2.imshow("Red mask", red_mask)
        cv2.imshow("White mask", white_mask)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break

    cap.release()
    if servo:
        servo.shutdown()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()