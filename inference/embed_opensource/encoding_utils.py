"""Re-export encoding utilities from the main chatts package."""
import importlib.util
import os
import sys

_chatts_path = os.path.join(os.path.dirname(__file__), "..", "chatts", "encoding_utils.py")
_spec = importlib.util.spec_from_file_location("chatts_encoding_utils", _chatts_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

timeseries_encoding = _mod.timeseries_encoding
sp_encoding = _mod.sp_encoding
minmax_scale_encoding = _mod.minmax_scale_encoding
