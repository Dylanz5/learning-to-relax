from __future__ import annotations
 
 import os
 from pathlib import Path
 
 
 def ensure_plots_dir(path: str | os.PathLike = "plots") -> Path:
     p = Path(path)
     p.mkdir(parents=True, exist_ok=True)
     return p
 
