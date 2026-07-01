"""
AutoGridSplit — detect the seams in a stitched grid / carousel / collage image
and split it into its individual panels. No model, CPU, any resolution.

Uses recursive guillotine decomposition, so it handles regular grids, uneven
grids, AND irregular ("bento") collages where different regions have different
split lines — as well as seamless (gutterless) grids.
"""
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


NODE_CLASS_MAPPINGS = {"AutoGridSplit": AutoGridSplit}
NODE_DISPLAY_NAME_MAPPINGS = {"AutoGridSplit": "Auto Grid / Carousel Split"}
