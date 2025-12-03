#!/usr/bin/env python3
import os
import re
import cv2
import time
import argparse

def parse_device(dev):
    # Принимает "0" или "/dev/video0"
    if isinstance(dev, int):
        return dev
    s = str(dev)
    if s.isdigit():
        return int(s)
    m = re.search(r"/dev/video(\d+)", s)
    if m:
        return int(m.group(1))
    return 0

def open_cam(dev_index, width, height, fps):
    backend = getattr(cv2, "CAP_V4L2", 0)
    cap = cv2.VideoCapture(dev_index, backend)
    if not cap.isOpened():
        return None
    # Попробуем MJPG (обычно доступно у C270)
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    except Exception:
        pass
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    time.sleep(0.2)
    ok, _ = cap.read()
    if not ok:
        cap.release()
        return None
    return cap

def main():
    parser = argparse.ArgumentParser(description="Тест камеры: сохранение кадров без окон")
    parser.add_argument("-d", "--device", default="0", help="Индекс камеры или /dev/videoN (по умолчанию 0)")
    parser.add_argument("--width",  type=int, default=1280, help="Ширина кадра (по умолчанию 1280)")
    parser.add_argument("--height", type=int, default=720,  help="Высота кадра (по умолчанию 720)")
    parser.add_argument("--fps",    type=int, default=30,   help="FPS запроса (по умолчанию 30)")
    parser.add_argument("--out",    default="/tmp/target.jpg", help="Путь вывода файла или папки")
    parser.add_argument("--every",  type=float, default=5.0, help="Интервал сохранения в секундах. 0 = сохранить один раз и выйти")
    parser.add_argument("--unique", action="store_true", help="Сохранять в папку с уникальными именами (out должен быть папкой)")
    parser.add_argument("--quality", type=int, default=90, help="JPEG качество 1..100 (по умолчанию 90)")
    args = parser.parse_args()

    dev_index = parse_device(args.device)
    cap = open_cam(dev_index, args.width, args.height, args.fps)
    if cap is None:
        print(f"[ERR] Не удалось открыть камеру {args.device}. Проверьте подключение и выполните: v4l2-ctl --list-devices")
        return

    if args.unique:
        out_dir = args.out
        os.makedirs(out_dir, exist_ok=True)
        print(f"[INFO] Сохранение каждые {args.every}s в папку: {out_dir} (уникальные имена)")
    else:
        # Если указали путь с папкой — создадим её
        out_dir = os.path.dirname(args.out) or "."
        if out_dir and out_dir not in (".", ""):
            os.makedirs(out_dir, exist_ok=True)
        print(f"[INFO] Сохранение каждые {args.every}s в файл: {args.out} (перезапись)")

    last_save = 0.0
    save_once = args.every <= 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[ERR] Кадр не получен.")
                break

            now = time.time()
            need_save = (save_once and last_save == 0) or (not save_once and (now - last_save) >= args.every)

            if need_save:
                last_save = now
                # Имя файла
                if args.unique:
                    fname = f"frame_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                    out_path = os.path.join(args.out, fname)
                else:
                    out_path = args.out

                # Сохраняем JPEG с заданным качеством
                ok = cv2.imwrite(out_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(args.quality)])
                if ok:
                    try:
                        size_kb = os.path.getsize(out_path) / 1024.0
                    except OSError:
                        size_kb = 0.0
                    h, w = frame.shape[:2]
                    print(f"[SAVE] OK {out_path} | {w}x{h} | {size_kb:.1f} KiB | {time.strftime('%H:%M:%S')}")
                else:
                    print(f"[SAVE] FAIL {out_path} | {time.strftime('%H:%M:%S')}")

                if save_once:
                    break

            # Небольшая пауза, чтобы не грузить CPU впустую
            time.sleep(0.01)

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        print("[INFO] Завершено.")

if __name__ == "__main__":
    main()