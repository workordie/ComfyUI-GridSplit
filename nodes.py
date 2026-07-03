"""
ComfyUI-GridSplit nodes.

  AutoGridSplit — detect the seams in a stitched grid/carousel/collage and split
                  it into its individual panels.
  GridStitch    — the inverse: repeat one image into an R x C grid, scaled to a
                  target total megapixels (no distortion; cell aspect = image aspect).

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
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "rows": ("INT", {"default": 3, "min": 1, "max": 32}),
                "cols": ("INT", {"default": 3, "min": 1, "max": 32}),
                "megapixels": ("FLOAT", {"default": 4.0, "min": 0.1, "max": 100.0, "step": 0.1,
                                         "tooltip": "Target TOTAL size of the stitched grid, in megapixels."}),
                "scale_method": (["nearest-exact", "bilinear", "area", "bicubic", "lanczos"],
                                 {"default": "bicubic",
                                  "tooltip": "Resampling filter (same set as ComfyUI's Upscale Image). bicubic/lanczos = sharp; area = clean downscale; nearest-exact = blocky."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("grid", "width", "height")
    FUNCTION = "stitch"
    CATEGORY = "image/grid"

    def stitch(self, image, rows, cols, megapixels, scale_method):
        B, H, W, C = image.shape
        aspect = W / H
        target_px = max(1.0, megapixels * 1_000_000.0)
        # aspect-preserving cell whose rows*cols copies tile to ~target_px
        h_c = max(1, int(round((target_px / (rows * cols * aspect)) ** 0.5)))
        w_c = max(1, int(round(aspect * h_c)))
        # resize each cell with ComfyUI's own upscaler (identical to the Upscale Image node)
        samples = image.movedim(-1, 1)  # [B,H,W,C] -> [B,C,H,W]
        resized = comfy.utils.common_upscale(samples, w_c, h_c, scale_method, "disabled")
        resized = resized.movedim(1, -1).clamp(0.0, 1.0)  # -> [B,h_c,w_c,C]
        out = torch.empty(1, rows * h_c, cols * w_c, C, dtype=image.dtype, device=image.device)
        for r in range(rows):
            for c in range(cols):
                idx = (r * cols + c) % B          # batch=1 -> repeat; batch>1 -> fill cells cyclically
                out[0, r * h_c:(r + 1) * h_c, c * w_c:(c + 1) * w_c, :] = resized[idx]
        return (out, cols * w_c, rows * h_c)


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
    cells: row-major list (len rows*cols) of [C,H,W] tensors or None."""
    def resize(chw, h, w):
        return comfy.utils.common_upscale(chw.unsqueeze(0), w, h, scale_method, "disabled")[0]

    row_strips = []
    for r in range(rows):
        rc = cells[r * cols:(r + 1) * cols]
        anchor_h = next((c.shape[1] for c in rc if c is not None), 512)
        pieces = []
        for c in rc:
            if c is None:                        # empty cell -> square placeholder at row height
                pieces.append(torch.full((3, anchor_h, anchor_h), float(empty_color)))
            else:
                h, w = c.shape[1], c.shape[2]
                pieces.append(resize(c, anchor_h, max(1, round(w * anchor_h / h))))
        row_strips.append(torch.cat(_match_channels(pieces), dim=2))   # concat along width

    anchor_w = row_strips[0].shape[2]
    finals = []
    for rs in row_strips:
        h, w = rs.shape[1], rs.shape[2]
        finals.append(resize(rs, max(1, round(h * anchor_w / w)), anchor_w))
    return torch.cat(_match_channels(finals), dim=1)                   # concat along height -> [C,H,W]


class GridStitchAdvanced:
    """Stitch multiple (different-size) images into an R x C grid, ImageStitch-style,
    scaled to a target megapixels. image_i -> cell i (row-major); empty cells -> black."""
    MAX_IMAGES = 16

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rows": ("INT", {"default": 2, "min": 1, "max": 8}),
                "cols": ("INT", {"default": 2, "min": 1, "max": 8}),
                "megapixels": ("FLOAT", {"default": 4.0, "min": 0.1, "max": 100.0, "step": 0.1,
                                         "tooltip": "Target TOTAL size of the stitched grid."}),
                "scale_method": (["lanczos", "bicubic", "area", "bilinear", "nearest-exact"],
                                 {"default": "lanczos"}),
            },
            "optional": {f"image_{i}": ("IMAGE",) for i in range(1, cls.MAX_IMAGES + 1)},
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("grid", "width", "height")
    FUNCTION = "stitch"
    CATEGORY = "image/grid"

    def stitch(self, rows, cols, megapixels, scale_method, **kwargs):
        n = rows * cols
        cells = []
        for i in range(1, n + 1):
            img = kwargs.get(f"image_{i}")
            cells.append(_to_chw(img) if img is not None else None)
        grid = stitch_imagestitch_style(cells, rows, cols, scale_method)
        gh, gw = grid.shape[1], grid.shape[2]
        scale = (max(1.0, megapixels * 1_000_000.0) / (gh * gw)) ** 0.5
        out_h, out_w = max(1, round(gh * scale)), max(1, round(gw * scale))
        grid = comfy.utils.common_upscale(grid.unsqueeze(0), out_w, out_h, scale_method, "disabled")[0]
        out = grid.movedim(0, -1).unsqueeze(0).clamp(0.0, 1.0)
        return (out, out_w, out_h)


NODE_CLASS_MAPPINGS = {
    "AutoGridSplit": AutoGridSplit,
    "GridStitch": GridStitch,
    "GridStitchAdvanced": GridStitchAdvanced,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoGridSplit": "Auto Grid / Carousel Split",
    "GridStitch": "Grid Stitch (repeat → grid)",
    "GridStitchAdvanced": "Grid Stitch Advanced (multi-image)",
}
