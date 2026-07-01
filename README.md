# ComfyUI-GridSplit

Auto-detect how a stitched image is divided and split it into its individual
panels. **No model, no weights, CPU, any resolution.**

Handles three kinds of layout automatically:

| Layout | example | how |
|---|---|---|
| **Regular grid** | 2×2 of variations | global seam projection |
| **Uneven grid** | 3 side-by-side, unequal widths | global seam projection |
| **Irregular collage** ("bento") | different split lines per region, a cell split again | recursive guillotine on the gutters |

It auto-routes: if the image has solid separator bands (black/white/any flat
gutter) it uses recursive guillotine decomposition (each region finds its own
local cuts, so irregular layouts work). If the panels are **seamless** (abut with
no gutter) it uses global row×column seam detection. Content edges (faces, text,
etc.) never trigger a split.

Validated: 2048² 2×2 → 4 tiles · 2728×1536 uneven 1×3 → 928/976/824 tiles ·
3344×1880 irregular 10-panel collage (incl. a column that's itself split in two).

## Node: `Auto Grid / Carousel Split`

**Inputs**
| name | what |
|------|------|
| `image` | the stitched grid / carousel / collage |
| `sensitivity` | default 1 (standard). **Higher = splits more eagerly / catches weaker seamless seams.** Only affects the seamless path. |
| `min_panel_px` | reject any split that would make a panel smaller than this |
| `edge_trim` | shave N px off each panel edge (kills blend / halo / anti-aliased gutter fringe) |

**Outputs**
| name | what |
|------|------|
| `panels` | **list** of images (tiles keep their native size) → feed into SaveImage to write N files |
| `count` | number of panels detected |
| `preview` | the input with detected panel boxes drawn in red — for visual QA |

`panels` is a list because panels have different sizes and can't share one batch tensor.
A non-grid image passes through unchanged as a single panel.

## Install
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/workordie/ComfyUI-GridSplit.git
# restart ComfyUI
```
No dependencies beyond torch/numpy (already in ComfyUI).

## Example
Drag [`example_workflows/autogrid_example.json`](example_workflows/autogrid_example.json)
onto the ComfyUI canvas — it generates an image and splits it, wiring `panels`
into SaveImage and `preview` into PreviewImage.
