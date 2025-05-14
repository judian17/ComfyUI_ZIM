# __init__.py for Combined ZIM Nodes Package

# Import mappings from zim_nodes.py
# These contain the ZimSegment node
try:
    from .zim_nodes import NODE_CLASS_MAPPINGS as ZIM_SEGMENT_CLASS_MAPPINGS, \
                           NODE_DISPLAY_NAME_MAPPINGS as ZIM_SEGMENT_DISPLAY_MAPPINGS
    print("Successfully imported mappings from zim_nodes.py")
except ImportError as e:
    print(f"Warning: Could not import from .zim_nodes: {e}. ZimSegment node might not be available.")
    ZIM_SEGMENT_CLASS_MAPPINGS = {}
    ZIM_SEGMENT_DISPLAY_MAPPINGS = {}
except Exception as e:
    print(f"Error importing from .zim_nodes: {e}. ZimSegment node might not be available.")
    ZIM_SEGMENT_CLASS_MAPPINGS = {}
    ZIM_SEGMENT_DISPLAY_MAPPINGS = {}


# Import mappings from zim_mask_preprocessing_node.py
# These contain MaskToPoints_ZIM and MaskToBbox_ZIM nodes
try:
    from .zim_mask_preprocessing_node import NODE_CLASS_MAPPINGS as MASK_PREPROC_CLASS_MAPPINGS, \
                                           NODE_DISPLAY_NAME_MAPPINGS as MASK_PREPROC_DISPLAY_MAPPINGS
    print("Successfully imported mappings from zim_mask_preprocessing_node.py")
except ImportError as e:
    print(f"Warning: Could not import from .zim_mask_preprocessing_node: {e}. Mask preprocessing nodes might not be available.")
    MASK_PREPROC_CLASS_MAPPINGS = {}
    MASK_PREPROC_DISPLAY_MAPPINGS = {}
except Exception as e:
    print(f"Error importing from .zim_mask_preprocessing_node: {e}. Mask preprocessing nodes might not be available.")
    MASK_PREPROC_CLASS_MAPPINGS = {}
    MASK_PREPROC_DISPLAY_MAPPINGS = {}

# Combine NODE_CLASS_MAPPINGS
NODE_CLASS_MAPPINGS = {}
NODE_CLASS_MAPPINGS.update(ZIM_SEGMENT_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(MASK_PREPROC_CLASS_MAPPINGS)

# Combine NODE_DISPLAY_NAME_MAPPINGS
NODE_DISPLAY_NAME_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS.update(ZIM_SEGMENT_DISPLAY_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(MASK_PREPROC_DISPLAY_MAPPINGS)

# Expose the combined mappings for ComfyUI to discover
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

if NODE_CLASS_MAPPINGS:
    print(f"### Loading: Combined ZIM Nodes Package ({len(NODE_CLASS_MAPPINGS)} nodes) ###")
    print(f"    Available class mappings: {list(NODE_CLASS_MAPPINGS.keys())}")
else:
    print("### Warning: Combined ZIM Nodes Package loaded, but no nodes were successfully mapped. Check import errors above. ###")
