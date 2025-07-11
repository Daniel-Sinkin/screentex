"""extractor.py
Annotate LaTeX regions in lecture slides and run a simulated OCR worker.

CLI
---
python extractor.py slides.pdf -o latex_regions

Positional arguments
--------------------
pdf               Path to the PDF file to annotate (default: slides.pdf)

Optional arguments
------------------
-o, --out         Output directory where image crops and OCR .tex files are stored
                  (default: ./latex_regions)

Key bindings inside the Slide Viewer window
------------------------------------------
click-drag : draw a box
u          : undo last box
q          : save boxes & next slide
b          : save boxes & back one slide
c          : clear all boxes on current slide
Esc        : quit program
"""

# pylint: disable=no-member
# pylint: disable=no-name-in-module

from __future__ import annotations

import argparse
import itertools
import time
from collections.abc import Buffer
from dataclasses import dataclass
from multiprocessing import Event, Process
from multiprocessing.synchronize import Event as EventT
from pathlib import Path
from typing import Any, Final, Iterator, cast

import cv2
import fitz
import numpy as np
from numpy.typing import NDArray

ESCAPE_KEY = 27


@dataclass(frozen=True)
class ViewerConfig:
    """All the window-related constants."""

    window_name: str = "Slide Viewer"
    window_position: tuple[int, int] = (100, 100)
    title_format: str = "{name} - ({current} / {total})"
    rect_color: tuple[int, int, int] = (0, 255, 0)
    rect_thickness: int = 2


@dataclass(frozen=True)
class KeyBindings:
    """Key bindings used in the viewer."""

    next_slide: int = ord("q")
    prev_slide: int = ord("b")
    undo_box: int = ord("u")
    clear_boxes: int = ord("c")
    quit: int = ESCAPE_KEY


class PixMap:  # pylint: disable=too-few-public-methods
    """Wrapper for fritz.pixmap"""

    def tobytes(self, x: str) -> Buffer:  # noqa: D401
        """Converts the pixmap to a buffer of the provided type."""
        raise NotImplementedError


_LATEX_SNIPPETS: Final[list[str]] = [
    r"\hat{y}=\sigma(Wx+b)",
    r"L=\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat{y}_i)^2",
    r"p(z\mid x)=\frac{p(x\mid z)p(z)}{p(x)}",
    r"\theta \leftarrow \theta-\eta\nabla_\theta L",
    r"q(z) \approx p(z \mid x)",
    r"\mathrm{ELBO}=\mathbb{E}_{q}[\log p(x,z)]-\mathbb{E}_{q}[\log q(z)]",
    r"K(x_i,x_j)=\exp\left(-\frac{\|x_i-x_j\|^2}{2\sigma^2}\right)",
    r"a^{(l)}=\mathrm{ReLU}(W^{(l)}a^{(l-1)}+b^{(l)})",
    r"\text{softmax}(z)_k = \frac{e^{z_k}}{\sum_j e^{z_j}}",
    r"f(x)=\mathrm{sign}(w^Tx+b)",
]

_LATEX_CYCLE: Final[Iterator[str]] = itertools.cycle(_LATEX_SNIPPETS)


def ocr_worker(folder: Path, stop_event: EventT) -> None:  # noqa: D401
    """Simulates an OCR service by watching *folder* for new PNG files and writing LaTeX output."""
    print("[OCR] Worker started")
    while True:
        if stop_event.is_set():
            break
        if not folder.exists():
            time.sleep(2)
            continue
        work_found = False
        for png_path in folder.glob("*.png"):
            if stop_event.is_set():
                break
            tex_path = png_path.with_suffix(".tex")
            if tex_path.exists():
                continue
            work_found = True
            latex_eq = next(_LATEX_CYCLE)
            print(f"[OCR] Processing {png_path.name} -> '{latex_eq}'")
            for _ in range(30):
                if stop_event.is_set():
                    break
                time.sleep(0.1)
            if stop_event.is_set():
                break
            tex_path.write_text(f"{latex_eq}\n")
            print(f"[OCR]   -> wrote {tex_path.name}")
        if not work_found:
            time.sleep(1)
    print("[OCR] Worker shutting down")


class BoxDrawer:
    """Handles bounding-box annotation for a single slide image."""

    def __init__(
        self, image: NDArray[np.uint8], slide_num: int, total_slides: int
    ) -> None:
        """Initializes the drawer with an image and slide metadata."""
        self.window_name: Final[str] = ViewerConfig.window_name
        self.original: NDArray[np.uint8] = image
        self.boxes: list[tuple[tuple[int, int], tuple[int, int]]] = []
        self.slide_num: int = slide_num
        self.total_slides: int = total_slides
        self._start: tuple[int, int] | None = None
        cv2.setMouseCallback(self.window_name, self._mouse_cb)

    def _mouse_cb(
        self, event: int, x: int, y: int, flags: int, _: Any
    ) -> None:  # noqa: D401
        """Mouse callback to record box start/end on drag events."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self._start = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self._start is not None:
            end = (x, y)
            self.boxes.append((self._start, end))
            self._start = None

    def clear_boxes(self) -> None:
        """Clears all drawn boxes."""
        self.boxes.clear()

    def run(
        self,
    ) -> tuple[str, list[tuple[tuple[int, int], tuple[int, int]]]]:  # noqa: D401
        """Displays the slide window and returns action ('next','back','quit') plus boxes."""
        while True:
            frame = self.original.copy()
            for pt1, pt2 in self.boxes:
                cv2.rectangle(
                    frame,
                    pt1,
                    pt2,
                    ViewerConfig.rect_color,
                    ViewerConfig.rect_thickness,
                )
            title = ViewerConfig.title_format.format(
                name=self.window_name, current=self.slide_num, total=self.total_slides
            )
            cv2.setWindowTitle(self.window_name, title)
            cv2.imshow(self.window_name, frame)
            key = cv2.waitKey(1)
            if key == KeyBindings.next_slide:
                return "next", self.boxes
            if key == KeyBindings.prev_slide:
                return "back", self.boxes
            if key == KeyBindings.quit:
                return "quit", []
            if key == KeyBindings.undo_box and len(self.boxes) > 0:
                self.boxes.pop()
            if key == KeyBindings.clear_boxes:
                self.clear_boxes()


def save_crops(
    img: NDArray[np.uint8],
    boxes: list[tuple[tuple[int, int], tuple[int, int]]],
    slide_idx: int,
    out_dir: Path,
) -> None:
    """Save image crops for the given slide based on annotated boxes."""
    h, w, _ = img.shape
    for j, (pt1, pt2) in enumerate(boxes, start=1):
        x1, y1 = pt1
        x2, y2 = pt2
        x1, x2 = sorted((max(0, x1), min(w, x2)))
        y1, y2 = sorted((max(0, y1), min(h, y2)))
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crop_path = out_dir.joinpath(f"slide_{slide_idx + 1:03}_crop_{j}.png")
        cv2.imwrite(str(crop_path), crop)
        print(f"[GUI] Saved {crop_path.name}")


def annotate_pdf(pdf_path: Path, out_dir: Path) -> None:
    """Launches GUI for annotating slides from *pdf_path* and saving crops to *out_dir*."""
    doc = fitz.open(pdf_path)
    n_slides = len(doc)
    cv2.namedWindow(ViewerConfig.window_name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(ViewerConfig.window_name, *ViewerConfig.window_position)
    slide_idx = 0

    while 0 <= slide_idx < n_slides:
        page = doc[slide_idx]
        pix = cast(PixMap, page.get_pixmap(dpi=200))
        img_data = np.frombuffer(pix.tobytes("png"), dtype=np.uint8)
        img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("Failed to decode slide image.")
        img = np.asarray(img, dtype=np.uint8)
        drawer = BoxDrawer(img, slide_idx + 1, n_slides)
        action, boxes = drawer.run()
        if action == "quit":
            break
        if boxes:
            save_crops(img, boxes, slide_idx, out_dir)
        if action == "back" and slide_idx > 0:
            slide_idx -= 1
        elif action == "next":
            slide_idx += 1
    cv2.destroyAllWindows()


def main() -> None:
    """Launches the window and the background OCR worker after parsing the argV for filepath."""
    print(__doc__)
    parser = argparse.ArgumentParser(
        description="Annotate LaTeX regions in a PDF deck of slides."
    )
    parser.add_argument(
        "pdf",
        nargs="?",
        default="slides.pdf",
        help="Path to the PDF slide deck (default: slides.pdf)",
    )
    parser.add_argument(
        "-o",
        "--out",
        default="latex_regions",
        help="Directory to store image crops and OCR .tex files (default: ./latex_regions)",
    )
    args = parser.parse_args()
    pdf_file = Path(args.pdf).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    if not pdf_file.is_file():
        raise FileNotFoundError(f"PDF file not found: {pdf_file}")
    out_dir.mkdir(parents=True, exist_ok=True)
    stop_evt = Event()
    worker = Process(target=ocr_worker, args=(out_dir, stop_evt))
    worker.start()
    try:
        annotate_pdf(pdf_file, out_dir)
    finally:
        stop_evt.set()
        worker.join()
        print("All done. Bye!")


if __name__ == "__main__":
    main()
