#!/usr/bin/env python3
"""screencap.py

Capture one or more interactive region screenshots on macOS and convert them to
LaTeX using a single long-lived worker process.  Screenshots are saved to a
target directory (``~/Desktop`` by default).  Each screenshot is handed to an
OCR stub running in a separate process that watches the directory for new PNG
files and writes a ``.tex`` file with the detected equation.

Run
----
$ python screencap.py                 # save to ~/Desktop
$ python screencap.py ./my_output_dir # custom directory
"""

from __future__ import annotations

import argparse
import itertools
import subprocess
from datetime import datetime
from multiprocessing import Event, Process
from multiprocessing.synchronize import Event as EventT
from pathlib import Path
from typing import Final, Iterator

_EQUATIONS: Final[list[str]] = [
    r"\int_{-\infty}^{\infty} e^{-x^{2}} \, dx = \sqrt{\pi}",
    r"e^{i\pi} + 1 = 0",
    r"\sum_{n=1}^{\infty} \frac{1}{n^{2}} = \frac{\pi^{2}}{6}",
    r"\nabla \cdot \mathbf{E} = \frac{\rho}{\varepsilon_{0}}",
    r"\mathbf{F} = m\mathbf{a}",
    r"\lim_{n \to \infty} \left(1 + \frac{1}{n}\right)^n = e",
    r"\frac{d}{dx}\sin x = \cos x",
    r"\det(A) = \prod_{i}\lambda_i",
    r"f(x) = \frac{1}{\sqrt{2\pi\sigma^{2}}}e^{-\frac{(x-\mu)^{2}}{2\sigma^{2}}}",
    r"\oint_{\partial \Sigma} \mathbf{B}\cdot d\mathbf{l} = \mu_{0} I",
]

_LATEX_CYCLE: Final[Iterator[str]] = itertools.cycle(_EQUATIONS)


def take_screenshot(out_path: Path) -> None:
    """Launch the macOS ``screencapture`` utility to grab a region screenshot."""
    proc = subprocess.Popen(["screencapture", "-i", str(out_path)])
    proc.wait()
    if proc.returncode != 0 or not out_path.exists():
        raise RuntimeError("Screenshot cancelled or failed")


def ocr_worker(folder: Path, stop_event: EventT) -> None:
    """Watch *folder* for new ``.png`` files and emit ``.tex`` files.

    The worker exits cleanly when *stop_event* is set.
    """
    print("[OCR] Worker started", flush=True)
    while not stop_event.is_set():
        work_found = False
        for png_path in folder.glob("*.png"):
            if stop_event.is_set():
                break
            tex_path = png_path.with_suffix(".tex")
            if tex_path.exists():
                continue
            work_found = True
            latex = next(_LATEX_CYCLE)
            print(f"[OCR] Processing {png_path.name} → '{latex}'", flush=True)
            for _ in range(30):  # simulate slow OCR ~3 s
                if stop_event.is_set():
                    break
                stop_event.wait(0.1)
            if stop_event.is_set():
                break
            tex_path.write_text(f"{latex}\n", encoding="utf-8")
            print(f"[OCR]   → wrote {tex_path.name}", flush=True)
        if not work_found:
            stop_event.wait(1.0)
    print("[OCR] Worker shutting down", flush=True)


def main() -> None:
    """Parse CLI arguments, start the worker, and collect screenshots."""
    parser = argparse.ArgumentParser(description="Screenshot → LaTeX converter")
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=str(Path.home().joinpath("Desktop")),
        help="Directory to store screenshots and generated .tex files (default: ~/Desktop)",
    )
    args = parser.parse_args()

    target_dir = Path(args.output_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    stop_event = Event()
    worker = Process(target=ocr_worker, args=(target_dir, stop_event))
    worker.start()

    try:
        while True:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            img_path = target_dir.joinpath(f"screenshot_{timestamp}.png")
            print("Drag to select a region… (Esc to cancel, Ctrl-C to quit)")
            take_screenshot(img_path)
            print(f"Saved screenshot → {img_path}")
    except KeyboardInterrupt:
        print("Stopping - waiting for worker …")
    finally:
        stop_event.set()
        worker.join()
        print("Done.")


if __name__ == "__main__":
    main()
