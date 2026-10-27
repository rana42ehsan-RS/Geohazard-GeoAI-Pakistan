import os
import sys
import warnings

# silence the harmless GDAL/PROJ data warnings and the Earth Engine dataset deprecation notice
_share = os.path.join(sys.prefix, "Library", "share")
if os.path.isdir(os.path.join(_share, "gdal")):
    os.environ.setdefault("GDAL_DATA", os.path.join(_share, "gdal"))
if os.path.isdir(os.path.join(_share, "proj")):
    os.environ.setdefault("PROJ_LIB", os.path.join(_share, "proj"))
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
