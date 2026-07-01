"""
Seam/gutter detection for AutoGridSplit. Pure torch, no models.

Two regimes, auto-dispatched:

* GUTTERED collages (black / white / any solid separator bands) — possibly
  irregular "bento" layouts where regions have different split lines. Handled by
  recursive guillotine decomposition that cuts ONLY at solid uniform bands, so
  content edges never trigger a split.

* SEAMLESS grids (panels abut with no gutter) — always a regular grid. Handled by
  global row×column projection: split at the strong discontinuity seams.

An image is routed to the guttered path iff it contains at least one interior
solid full-span band; otherwise the seamless path.
"""
import torch


def _robust_z(x):
    med = x.median()
    mad = (x - med).abs().median()
    if mad < 1e-6:
        mad = x.std() / 1.4826 + 1e-6
    return (x - med) / (1.4826 * mad)


def _runs(mask):
    out, s = [], None
    for i, v in enumerate(mask):
        if v and s is None:
            s = i
        elif not v and s is not None:
            out.append((s, i - 1)); s = None
    if s is not None:
        out.append((s, len(mask) - 1))
    return out


# ----------------------------------------------------------------------------- gutter path
def _line_std(gray, axis):
    return gray.std(0) if axis == 1 else gray.std(1)   # axis 1 -> per column, 0 -> per row


def _gutter_runs(gray, axis, cfg):
    """Solid (near-flat) full-span bands on one axis: list of (start, end)."""
    length = gray.shape[axis]
    std = _line_std(gray, axis)
    flat = [std[i].item() < cfg["gutter_std_abs"] for i in range(length)]
    out = []
    for (s, e) in _runs(flat):
        if s <= 1 or e >= length - 2:                       # ignore outer frame
            continue
        w = e - s + 1
        if w < cfg["min_gutter"] or w > cfg["max_gutter_frac"] * length:
            continue
        c = (s + e) // 2
        if c < cfg["min_panel"] or (length - c) < cfg["min_panel"]:
            continue
        out.append((s, e))
    return out


def _has_gutters(gray, cfg):
    return bool(_gutter_runs(gray, 0, cfg) or _gutter_runs(gray, 1, cfg))


def _best_gutter(gray, cfg):
    best = None
    for axis in (1, 0):
        if gray.shape[axis] < 2 * cfg["min_panel"] + 1:
            continue
        for (s, e) in _gutter_runs(gray, axis, cfg):
            w = e - s + 1                                    # prefer the widest (most prominent) band
            if best is None or w > best[3]:
                best = (axis, (s, e), (s + e) // 2, w)
    return best


def _gutter_decompose(gray, y0, x0, cfg, depth, out):
    h, w = gray.shape
    cut = None if depth >= cfg["max_depth"] else _best_gutter(gray, cfg)
    if cut is None:
        out.append((y0, y0 + h, x0, x0 + w))
        return
    axis, (s, e), _c, _w = cut
    if axis == 1:                                           # vertical -> left | right (drop band)
        _gutter_decompose(gray[:, :s], y0, x0, cfg, depth + 1, out)
        _gutter_decompose(gray[:, e + 1:], y0, x0 + e + 1, cfg, depth + 1, out)
    else:                                                   # horizontal -> top / bottom (drop band)
        _gutter_decompose(gray[:s, :], y0, x0, cfg, depth + 1, out)
        _gutter_decompose(gray[e + 1:, :], y0 + e + 1, x0, cfg, depth + 1, out)


def _trim_frame(gray, cfg):
    """Crop a solid outer border. Returns (top, bottom, left, right)."""
    H, W = gray.shape
    rstd, cstd = gray.std(1), gray.std(0)
    a = cfg["gutter_std_abs"]
    top = 0
    while top < H - 1 and rstd[top].item() < a:
        top += 1
    bot = H
    while bot > top + 1 and rstd[bot - 1].item() < a:
        bot -= 1
    left = 0
    while left < W - 1 and cstd[left].item() < a:
        left += 1
    right = W
    while right > left + 1 and cstd[right - 1].item() < a:
        right -= 1
    return top, bot, left, right


# ----------------------------------------------------------------------------- seamless path
def _seams_1d(profile, length, cfg):
    z = _robust_z(profile)
    dl = profile.tolist(); zl = z.tolist()
    # 1) candidate local-max boundaries above the base floor
    cand = [i for i in range(1, len(dl) - 1)
            if zl[i] >= cfg["z_thresh"] and dl[i] >= max(dl[max(0, i - 3):i + 4]) - 1e-9]
    if not cand:
        return []
    # 2) relative suppression: drop edges far weaker than the dominant seam.
    #    True grid seams cluster near the top; content edges sit well below.
    cutoff = cfg["rel_frac"] * max(zl[i] for i in cand)
    gap = max(4, int(cfg["min_gap_frac"] * length))
    peaks = []
    for i in cand:
        if zl[i] < cutoff:
            continue
        if peaks and (i - peaks[-1]) < gap:                 # merge near-duplicates
            if dl[i] > dl[peaks[-1]]:
                peaks[-1] = i
            continue
        peaks.append(i)
    seams, prev = [], 0
    for p in peaks:
        pos = p + 1
        if pos - prev >= cfg["min_panel"] and (length - pos) >= cfg["min_panel"]:
            seams.append(pos); prev = pos
    return seams


def _seamless_split(gray, cfg):
    H, W = gray.shape
    dcol = (gray[:, 1:] - gray[:, :-1]).abs().mean(0)
    drow = (gray[1:, :] - gray[:-1, :]).abs().mean(1)
    cs = _seams_1d(dcol, W, cfg)
    rs = _seams_1d(drow, H, cfg)
    xs = [0] + cs + [W]
    ys = [0] + rs + [H]
    return [(ys[r], ys[r + 1], xs[c], xs[c + 1])
            for r in range(len(ys) - 1) for c in range(len(xs) - 1)]


# ----------------------------------------------------------------------------- public API
def find_panels(img_hwc, sensitivity=1.0, min_panel=64, min_gutter=3,
                max_gutter_frac=0.06, gutter_std_abs=0.01, min_gap_frac=0.05,
                rel_frac=0.3, max_depth=12):
    gray = img_hwc.float().mean(-1)
    cfg = {
        "z_thresh": max(3.0, 6.0 - sensitivity),
        "min_panel": int(min_panel),
        "min_gutter": int(min_gutter),
        "max_gutter_frac": float(max_gutter_frac),
        "gutter_std_abs": float(gutter_std_abs),   # only near-SOLID bands count as gutters
        "min_gap_frac": float(min_gap_frac),
        "rel_frac": float(rel_frac),               # seamless: keep seams >= rel_frac * strongest
        "max_depth": int(max_depth),
    }
    if _has_gutters(gray, cfg):
        top, bot, left, right = _trim_frame(gray, cfg)
        out = []
        _gutter_decompose(gray[top:bot, left:right], top, left, cfg, 0, out)
    else:
        out = _seamless_split(gray, cfg)
    out.sort(key=lambda b: (b[0], b[2]))
    return out


def slice_boxes(img_hwc, boxes, edge_trim=0):
    tiles = []
    H, W = img_hwc.shape[0], img_hwc.shape[1]
    for (y0, y1, x0, x1) in boxes:
        yy0 = y0 + (edge_trim if y0 > 0 else 0)
        yy1 = y1 - (edge_trim if y1 < H else 0)
        xx0 = x0 + (edge_trim if x0 > 0 else 0)
        xx1 = x1 - (edge_trim if x1 < W else 0)
        if yy1 > yy0 and xx1 > xx0:
            tiles.append(img_hwc[yy0:yy1, xx0:xx1, :])
    return tiles


def draw_preview(img_hwc, boxes, thickness=4):
    out = img_hwc.clone()
    H, W = out.shape[0], out.shape[1]
    red = torch.tensor([1.0, 0.0, 0.0], dtype=out.dtype, device=out.device)
    t = max(1, thickness)
    for (y0, y1, x0, x1) in boxes:
        out[max(0, y0):min(H, y0 + t), max(0, x0):min(W, x1), :] = red
        out[max(0, y1 - t):min(H, y1), max(0, x0):min(W, x1), :] = red
        out[max(0, y0):min(H, y1), max(0, x0):min(W, x0 + t), :] = red
        out[max(0, y0):min(H, y1), max(0, x1 - t):min(W, x1), :] = red
    return out
