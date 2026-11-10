"""colabgraphics: describe a rendering technique in a notebook, write its shaders in HLSL, run it with WebGPU."""
from .display import Snapshot, snapshot_html, standalone_html
from .technique import Mesh, Program, ShaderError, Technique, TechniqueError, Texture, Var

__all__ = ["Technique", "TechniqueError", "ShaderError", "Texture", "Mesh", "Var", "Program",
           "Snapshot", "snapshot_html", "standalone_html"]
__version__ = "0.0.1"
