"""colabgraphics: describe a rendering technique in a notebook, run it with WebGPU."""
from .display import Snapshot, snapshot_html, standalone_html
from .technique import Mesh, Program, Technique, TechniqueError, Texture, Var

__all__ = ["Technique", "TechniqueError", "Texture", "Mesh", "Var", "Program", "Snapshot", "snapshot_html", "standalone_html"]
__version__ = "0.0.1"
