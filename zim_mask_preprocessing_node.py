import torch
import numpy as np
import cv2  # ComfyUI typically includes opencv-python-headless
import json

class MaskToPoints:
    """
    A node that takes a mask and outputs the center points of its white areas.
    If there are multiple disconnected white areas, it outputs the center point for each.
    The output is a JSON string representing a list of points, e.g., [{"x": 100, "y": 150}, {"x": 200, "y": 250}].
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("points",) # Added for UI display name
    FUNCTION = "get_points_from_mask"
    CATEGORY = "ComfyUI_ZIM/mask"  # Changed category

    def get_points_from_mask(self, mask: torch.Tensor):
        if mask is None:
            return (json.dumps([]),)

        if mask.ndim == 3:
            mask_np = mask[0].cpu().numpy()
        elif mask.ndim == 2:
            mask_np = mask.cpu().numpy()
        else:
            print(f"MaskToPoints: Warning - Invalid mask dimensions: {mask.shape}. Expected 2 or 3 dimensions.")
            return (json.dumps([]),)

        if mask_np.ndim != 2:
            print(f"MaskToPoints: Warning - Mask numpy conversion resulted in unexpected dimensions: {mask_np.shape}")
            return (json.dumps([]),)

        if mask_np.max() <= 1.0 and mask_np.min() >= 0.0:
            processed_mask = (mask_np * 255).astype(np.uint8)
        else:
            processed_mask = np.clip(mask_np, 0, 255).astype(np.uint8)
        
        _, binary_mask = cv2.threshold(processed_mask, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        points_list = []
        if contours:
            for contour in contours:
                if cv2.contourArea(contour) > 0:
                    x, y, w, h = cv2.boundingRect(contour)
                    center_x = x + w / 2.0
                    center_y = y + h / 2.0
                    points_list.append({"x": int(round(center_x)), "y": int(round(center_y))})
        
        return (json.dumps(points_list),)

class MaskToBbox:
    """
    A node that takes a mask and outputs the bounding box of its white areas.
    If there are multiple disconnected white areas, it outputs a single bounding box
    that encompasses all white areas.
    The output is a BBOX tuple: (x_min, y_min, width, height).
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("BBOX",) # ComfyUI might interpret this as a generic tuple/list.
    # RETURN_NAMES = ("bbox",) # Optional
    FUNCTION = "get_bbox_from_mask"
    CATEGORY = "ComfyUI_ZIM/mask" # Changed category

    def get_bbox_from_mask(self, mask: torch.Tensor):
        if mask is None:
            return ((0, 0, 0, 0),) # Return a zero bbox if no mask

        # Convert mask tensor to NumPy array
        if mask.ndim == 3: # Batch, Height, Width
            mask_np = mask[0].cpu().numpy() # Process the first mask in the batch
        elif mask.ndim == 2: # Height, Width
            mask_np = mask.cpu().numpy()
        else:
            print(f"MaskToBbox: Warning - Invalid mask dimensions: {mask.shape}. Expected 2 or 3 dimensions.")
            return ((0, 0, 0, 0),)

        if mask_np.ndim != 2:
            print(f"MaskToBbox: Warning - Mask numpy conversion resulted in unexpected dimensions: {mask_np.shape}")
            return ((0, 0, 0, 0),)

        # Normalize and binarize the mask
        if mask_np.max() <= 1.0 and mask_np.min() >= 0.0: # Assuming mask is in 0.0-1.0 range
            processed_mask = (mask_np * 255).astype(np.uint8)
        else: # Assuming mask might be in 0-255 range already or other
            processed_mask = np.clip(mask_np, 0, 255).astype(np.uint8)
        
        _, binary_mask = cv2.threshold(processed_mask, 127, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return ((0, 0, 0, 0),) # No contours found

        # Calculate the overall bounding box for all contours
        overall_x_min = mask_np.shape[1] # Initialize with image width
        overall_y_min = mask_np.shape[0] # Initialize with image height
        overall_x_max = 0
        overall_y_max = 0

        for contour in contours:
            if cv2.contourArea(contour) > 0: # Consider only non-empty contours
                x, y, w, h = cv2.boundingRect(contour)
                overall_x_min = min(overall_x_min, x)
                overall_y_min = min(overall_y_min, y)
                overall_x_max = max(overall_x_max, x + w)
                overall_y_max = max(overall_y_max, y + h)
        
        if overall_x_max <= overall_x_min or overall_y_max <= overall_y_min:
             # This case can happen if contours were found but had zero area after filtering,
             # or if initial overall_x_min/max were not updated.
            return ((0,0,0,0),)

        bbox_x = overall_x_min
        bbox_y = overall_y_min
        bbox_width = overall_x_max - overall_x_min
        bbox_height = overall_y_max - overall_y_min
        
        return ((int(bbox_x), int(bbox_y), int(bbox_width), int(bbox_height)),)

# NODE CLASS MAPPINGS
NODE_CLASS_MAPPINGS = {
    "MaskToPoints_ZIM": MaskToPoints,  # Renamed to avoid potential conflicts if old file is still around
    "MaskToBbox_ZIM": MaskToBbox
}

# NODE DISPLAY NAME MAPPINGS
NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskToPoints_ZIM": "Mask To Points (ZIM)",
    "MaskToBbox_ZIM": "Mask To Bbox (ZIM)"
}
