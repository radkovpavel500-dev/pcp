#!/usr/bin/env python3
import cv2
import numpy as np
import time

def open_c270_v4l2(preferred_index=None, width=1280, height=720, fps=30):
    indices = [preferred_index] if isinstance(preferred_index, int) else [0, 1, 2, 3, 4]
    for idx in indices:
        try:
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2) if hasattr(cv2, "CAP_V4L2") else cv2.VideoCapture(idx)
            if not cap.isOpened():
                continue
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

def main():
    cap, desc = open_c270_v4l2()
    if cap is None:
        print("Не удалось открыть Logitech C270.")
        return

    win = "Camera only test"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    # Жёстко заданные пороги (можно при необходимости подстроить)
    red1 = (0, 120, 60), (10, 255, 255)
    red2 = (170, 120, 60), (179, 255, 255)
    white = (0, 0, 200), (179, 60, 255)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Кадр не получен.")
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        r1 = cv2.inRange(hsv, np.array(red1[0],np.uint8), np.array(red1[1],np.uint8))
        r2 = cv2.inRange(hsv, np.array(red2[0],np.uint8), np.array(red2[1],np.uint8))
        red_mask = cv2.bitwise_or(r1, r2)
        white_mask = cv2.inRange(hsv, np.array(white[0],np.uint8), np.array(white[1],np.uint8))

        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN,  kernel, iterations=1)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cx, cy = None, None
        wmax, hmax = 0, 0
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            if cv2.contourArea(cnt) > 1000:
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect).astype(int)
                xs, ys = box[:,0], box[:,1]
                x_min, x_max = xs.min(), xs.max()
                y_min, y_max = ys.min(), ys.max()
                cx, cy   = int((x_min + x_max)/2), int((y_min + y_max)/2)
                wmax, hmax = int(x_max-x_min), int(y_max-y_min)

                # Быстрая проверка "белого внутри"
                poly_mask = np.zeros_like(white_mask)
                cv2.fillPoly(poly_mask, [box], 255)
                area_in = cv2.countNonZero(poly_mask)
                if area_in > 0:
                    white_in = cv2.countNonZero(cv2.bitwise_and(white_mask, poly_mask))
                    ratio = 100.0 * white_in / area_in
                else:
                    ratio = 0

                if ratio > 5:  # простая валидация
                    cv2.polylines(frame, [box], True, (0,255,255), 2)
                    cv2.circle(frame, (cx, cy), 6, (0,255,0), -1)
                else:
                    cx, cy = None, None
                    wmax, hmax = 0, 0

        h, w = frame.shape[:2]
        txt = f"Center: { (cx, cy) if cx is not None else '—' } | W:{wmax}px H:{hmax}px"
        cv2.rectangle(frame, (0, h-30), (w, h), (0,0,0), thickness=cv2.FILLED)
        cv2.putText(frame, txt, (10, h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0,255,0) if cx is not None else (0,0,255), 2, cv2.LINE_AA)

        cv2.imshow(win, frame)
        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()