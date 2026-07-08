"""
ComfyUI-GridSplit nodes.

  AutoGridSplit — detect the seams in a stitched grid/carousel/collage and split
                  it into its individual panels.
  GridStitch    — the inverse: repeat one image into an R x C grid, scaled to a
                  target total megapixels (no distortion; cell aspect = image aspect).
  GridStitchAdvanced — stitch many images into an R x C grid, and optionally emit a
                  per-cell MASK (which cells denoise vs stay preserved) for masked
                  latent workflows (SetLatentNoiseMask / inpaint / partial denoise).

No model, CPU, any resolution.
"""
import torch
import comfy.utils
from .gridsplit import find_panels, slice_boxes, draw_preview


class AutoGridSplit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "sensitivity": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 10.0, "step": 0.5,
                                          "tooltip": "Splitting aggressiveness. 1 = standard (recommended). Higher = split more eagerly / catch weaker seams."}),
                "min_panel_px": ("INT", {"default": 64, "min": 8, "max": 8192,
                                         "tooltip": "Reject any split that would make a panel smaller than this."}),
                "edge_trim": ("INT", {"default": 0, "min": 0, "max": 256,
                                      "tooltip": "Shave N px off each interior seam edge (kills blend/halo/leftover gutter)."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "IMAGE")
    RETURN_NAMES = ("panels", "count", "preview")
    OUTPUT_IS_LIST = (True, False, False)   # panels is a list (tiles differ in size); count/preview are scalars
    FUNCTION = "split"
    CATEGORY = "image/grid"

    def split(self, image, sensitivity, min_panel_px, edge_trim):
        panels_out = []
        preview = image[0:1]
        for b in range(image.shape[0]):
            img = image[b]  # [H,W,C]
            boxes = find_panels(img, sensitivity=sensitivity, min_panel=min_panel_px)
            tiles = slice_boxes(img, boxes, edge_trim)
            panels_out.extend(t.unsqueeze(0) for t in tiles)
            if b == 0:
                preview = draw_preview(img, boxes).unsqueeze(0)

        if not panels_out:                       # nothing detected -> pass the image through unchanged
            panels_out = [image[0:1]]
        return (panels_out, len(panels_out), preview)


class GridStitch:
    """Single image -> R x C grid, same engine as the advanced node
    (ImageStitch-style), scaled to a target megapixels. The one image fills
    every cell. It's the multi-image node with a single input, for quick
    single-image grids."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "rows": ("INT", {"default": 2, "min": 1, "max": 4}),
                "cols": ("INT", {"default": 2, "min": 1, "max": 4}),
                "megapixels": ("FLOAT", {"default": 4.0, "min": 0.1, "max": 100.0, "step": 0.1,
                                         "tooltip": "Target TOTAL size of the stitched grid, in megapixels."}),
                "scale_method": (["lanczos", "bicubic", "area", "bilinear", "nearest-exact"],
                                 {"default": "lanczos"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("grid", "width", "height")
    FUNCTION = "stitch"
    CATEGORY = "image/grid"

    def stitch(self, image, rows, cols, megapixels, scale_method):
        cell = _to_chw(image)                       # one image -> every cell
        cells = [cell] * (rows * cols)
        grid, _boxes = stitch_imagestitch_style(cells, rows, cols, scale_method)
        gh, gw = grid.shape[1], grid.shape[2]
        scale = (max(1.0, megapixels * 1_000_000.0) / (gh * gw)) ** 0.5
        out_h, out_w = max(1, round(gh * scale)), max(1, round(gw * scale))
        grid = comfy.utils.common_upscale(grid.unsqueeze(0), out_w, out_h, scale_method, "disabled")[0]
        out = grid.movedim(0, -1).unsqueeze(0).clamp(0.0, 1.0)
        return (out, out_w, out_h)


def _to_chw(img):
    """[B,H,W,C] or [H,W,C] -> [C,H,W]."""
    if img.dim() == 4:
        img = img[0]
    return img.movedim(-1, 0)


def _match_channels(pieces):
    c = max(p.shape[0] for p in pieces)
    out = []
    for p in pieces:
        if p.shape[0] < c:                       # pad missing channels with 1.0 (like ImageStitch)
            pad = torch.ones(c - p.shape[0], p.shape[1], p.shape[2], dtype=p.dtype, device=p.device)
            p = torch.cat([p, pad], dim=0)
        out.append(p)
    return out


def stitch_imagestitch_style(cells, rows, cols, scale_method="lanczos", empty_color=0.0):
    """Replicates ComfyUI ImageStitch (match_image_size=True) over an R x C grid:
    per row, resize each image to the row's first-image HEIGHT (aspect-preserved);
    then resize each row strip to the first row's WIDTH. No padding -> clean rectangle.
    cells: row-major list (len rows*cols) of [C,H,W] tensors or None.
    Returns (grid [C,H,W], boxes) where boxes is a row-major list of
    (y0, y1, x0, x1) cell rectangles in the returned grid's pixel space."""
    def resize(chw, h, w):
        return comfy.utils.common_upscale(chw.unsqueeze(0), w, h, scale_method, "disabled")[0]

    row_strips = []
    row_widths = []                              # per-row list of the pre-normalization cell widths
    for r in range(rows):
        rc = cells[r * cols:(r + 1) * cols]
        anchor_h = next((c.shape[1] for c in rc if c is not None), 512)
        pieces, widths = [], []
        for c in rc:
            if c is None:                        # empty cell -> square placeholder at row height
                pieces.append(torch.full((3, anchor_h, anchor_h), float(empty_color)))
                widths.append(anchor_h)
            else:
                h, w = c.shape[1], c.shape[2]
                pw = max(1, round(w * anchor_h / h))
                pieces.append(resize(c, anchor_h, pw))
                widths.append(pw)
        row_strips.append(torch.cat(_match_channels(pieces), dim=2))   # concat along width
        row_widths.append(widths)

    anchor_w = row_strips[0].shape[2]
    finals, boxes, y = [], [], 0
    for r, rs in enumerate(row_strips):
        h, w = rs.shape[1], rs.shape[2]
        fh = max(1, round(h * anchor_w / w))
        finals.append(resize(rs, fh, anchor_w))                        # normalize row strip to shared width
        widths = row_widths[r]
        total = float(sum(widths)) or 1.0
        cum = 0
        for c in range(cols):                                          # cell x-bounds within the normalized strip
            x0 = int(round(cum / total * anchor_w))
            cum += widths[c]
            x1 = int(round(cum / total * anchor_w))
            boxes.append((y, y + fh, x0, x1))
        y += fh
    grid = torch.cat(_match_channels(finals), dim=1)                   # concat along height -> [C,H,W]
    return grid, boxes


RATIOS = ["1:1", "3:4", "4:3", "2:3", "3:2", "4:5", "9:16", "16:9"]


def _parse_ratio(s):
    """'3:4' -> width/height aspect (0.75)."""
    try:
        w, h = s.split(":")
        return float(w) / float(h)
    except Exception:
        return 1.0


def _center_crop_to_aspect(chw, aspect):
    """Center-crop [C,H,W] so width/height == aspect (no scaling, just crop)."""
    _, h, w = chw.shape
    if w / h > aspect:                       # too wide -> trim the sides
        nw = max(1, int(round(h * aspect)))
        x0 = (w - nw) // 2
        return chw[:, :, x0:x0 + nw]
    nh = max(1, int(round(w / aspect)))      # too tall -> trim top/bottom
    y0 = (h - nh) // 2
    return chw[:, y0:y0 + nh, :]


def stitch_uniform(cells, rows, cols, aspect, scale_method, base_h=512):
    """Manual mode: center-crop every image to `aspect`, resize to one shared cell
    size, tile row-major into a clean rows x cols rectangle. Empty cells -> black.
    Returns (grid [C,H,W], boxes) — uniform cell rectangles, row-major."""
    ch = base_h
    cw = max(1, int(round(base_h * aspect)))
    ref = next((c for c in cells if c is not None), None)
    device = ref.device if ref is not None else "cpu"
    dtype = ref.dtype if ref is not None else torch.float32

    def resize(chw, h, w):
        return comfy.utils.common_upscale(chw.unsqueeze(0), w, h, scale_method, "disabled")[0]

    tiles = []
    for cell in cells:
        if cell is None:
            tiles.append(torch.zeros(3, ch, cw, device=device, dtype=dtype))
        else:
            tiles.append(resize(_center_crop_to_aspect(cell, aspect), ch, cw))
    tiles = _match_channels(tiles)
    strips = [torch.cat(tiles[r * cols:(r + 1) * cols], dim=2) for r in range(rows)]
    grid = torch.cat(strips, dim=1)          # [C, rows*ch, cols*cw]
    boxes = [(r * ch, (r + 1) * ch, c * cw, (c + 1) * cw)
             for r in range(rows) for c in range(cols)]
    return grid, boxes


def _parse_cells(spec, n):
    """'2,4-6' -> {2,4,5,6}, clamped to 1..n. Empty/garbage -> empty set."""
    out = set()
    if not spec:
        return out
    for part in str(spec).replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            try:
                a, b = part.split("-")
                a, b = int(a), int(b)
                for k in range(min(a, b), max(a, b) + 1):
                    if 1 <= k <= n:
                        out.add(k)
            except Exception:
                pass
        else:
            try:
                k = int(part)
                if 1 <= k <= n:
                    out.add(k)
            except Exception:
                pass
    return out


def _build_mask(boxes, gh, gw, out_h, out_w, denoise):
    """MASK [1,out_h,out_w]: white(1)=denoise, black(0)=preserve. `boxes` are in the
    pre-scale grid space (gh x gw); rescale to the final (out_h x out_w). An empty
    selection returns all-white (a wired-but-unconfigured mask is a no-op = denoise all)."""
    m = torch.zeros(out_h, out_w)
    if not denoise:
        m += 1.0
        return m.unsqueeze(0)
    fy = out_h / float(gh)
    fx = out_w / float(gw)
    for idx, (y0, y1, x0, x1) in enumerate(boxes, start=1):
        if idx in denoise:
            yy0 = max(0, min(out_h, int(round(y0 * fy))))
            yy1 = max(0, min(out_h, int(round(y1 * fy))))
            xx0 = max(0, min(out_w, int(round(x0 * fx))))
            xx1 = max(0, min(out_w, int(round(x1 * fx))))
            m[yy0:yy1, xx0:xx1] = 1.0
    return m.unsqueeze(0)


class GridStitchAdvanced:
    """Stitch multiple (different-size) images into an R x C grid, ImageStitch-style,
    scaled to a target megapixels. image_i -> cell i (row-major); empty cells -> black.
    Also emits a per-cell MASK for masked-latent workflows: `mask_cells` lists the cells
    to DENOISE (row-major, 1-indexed, e.g. '2,4-6'); the rest are preserved. The MASK
    lines up with the returned grid, so it can feed SetLatentNoiseMask directly."""
    MAX_IMAGES = 16

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rows": ("INT", {"default": 2, "min": 1, "max": 4}),
                "cols": ("INT", {"default": 2, "min": 1, "max": 4}),
                "mode": (["auto", "manual"], {"default": "auto",
                         "tooltip": "auto = preserve every image, no cropping (cells can differ). manual = center-crop every image to one ratio for a uniform grid."}),
                "ratio": (RATIOS, {"default": "3:4",
                          "tooltip": "Manual mode only: the aspect every image is center-cropped to before stitching."}),
                "megapixels": ("FLOAT", {"default": 4.0, "min": 0.1, "max": 100.0, "step": 0.1,
                                         "tooltip": "Target TOTAL size of the stitched grid."}),
                "scale_method": (["lanczos", "bicubic", "area", "bilinear", "nearest-exact"],
                                 {"default": "lanczos"}),
                "mask_cells": ("STRING", {"default": "",
                               "tooltip": "Cells to DENOISE for the MASK output (row-major, 1-indexed, e.g. '2,4-6'). Others are preserved. Empty = denoise all (mask is a no-op). Set via the picker's mask toggle, or by hand / from an app."}),
            },
            "optional": {f"image_{i}": ("IMAGE",) for i in range(1, cls.MAX_IMAGES + 1)},
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "MASK")
    RETURN_NAMES = ("grid", "width", "height", "mask")
    FUNCTION = "stitch"
    CATEGORY = "image/grid"

    def stitch(self, rows, cols, mode, ratio, megapixels, scale_method, mask_cells="", **kwargs):
        n = rows * cols
        cells = []
        for i in range(1, n + 1):
            img = kwargs.get(f"image_{i}")
            cells.append(_to_chw(img) if img is not None else None)
        if mode == "manual":
            grid, boxes = stitch_uniform(cells, rows, cols, _parse_ratio(ratio), scale_method)
        else:
            grid, boxes = stitch_imagestitch_style(cells, rows, cols, scale_method)
        gh, gw = grid.shape[1], grid.shape[2]
        scale = (max(1.0, megapixels * 1_000_000.0) / (gh * gw)) ** 0.5
        out_h, out_w = max(1, round(gh * scale)), max(1, round(gw * scale))
        grid = comfy.utils.common_upscale(grid.unsqueeze(0), out_w, out_h, scale_method, "disabled")[0]
        out = grid.movedim(0, -1).unsqueeze(0).clamp(0.0, 1.0)
        mask = _build_mask(boxes, gh, gw, out_h, out_w, _parse_cells(mask_cells, n))
        return (out, out_w, out_h, mask)


NODE_CLASS_MAPPINGS = {
    "AutoGridSplit": AutoGridSplit,
    "GridStitch": GridStitch,
    "GridStitchAdvanced": GridStitchAdvanced,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoGridSplit": "Auto Grid / Carousel Split",
    "GridStitch": "Auto Grid / Stitch (one image)",
    "GridStitchAdvanced": "Auto Grid / Stitch Advanced (multi-image)",
}
