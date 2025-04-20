# Import mappings from our single nodes file
from .zim_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Optionally add the web directory if your JS needs to be explicitly mentioned here
# Usually, the WEB_DIRECTORY in the node file is sufficient.
# WEB_DIRECTORY = "./js" 

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

print("Initializing ZIM Custom Nodes")
