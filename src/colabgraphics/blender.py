"""Let Blender do what it is good at: scenes and cameras.

Blender runs inside the notebook's Python as the `bpy` module (`pip install bpy`), with
no window. This module turns the current Blender scene into plain numpy arrays that a
technique can use, and asks Blender for answers to check a renderer against.

    scene()      world-space triangles (positions, normals, uvs, colours), the camera's
                 view and projection matrices in WebGPU's conventions, and the lights
    reference()  per-pixel depth, coverage and normals, from Blender's own ray casting:
                 independent of anything this project renders
    render()     the picture Blender's Cycles renderer makes of the same scene

Conventions. Blender's world is right-handed with Z up. A camera looks along its local
-Z with +Y up. Matrices are returned in the usual maths layout (they multiply column
vectors), which is how `Technique.var` takes them: in a shader, write
`mul(vars.view_projection, float4(position, 1))`. The projection maps depth to 0..1,
which is what WebGPU expects, not the -1..1 of OpenGL that Blender itself produces.
Images and reference arrays have row 0 at the top.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field

import numpy as np

# OpenGL clip space (z in -w..w) to WebGPU clip space (z in 0..w).
GL_TO_WEBGPU = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0.5, 0.5], [0, 0, 0, 1]], np.float64)
DEFAULT_COLOR = (0.8, 0.8, 0.8)


def _bpy():
    try:
        import bpy
    except ImportError:
        raise ImportError("Blender's Python module is not installed. Run: pip install bpy\n"
                          "(it needs the Python version that Blender itself ships with)") from None
    return bpy


@dataclass
class Camera:
    name: str
    view: np.ndarray             # world -> camera
    projection: np.ndarray       # camera -> clip, depth 0..1
    position: np.ndarray
    clip: tuple[float, float]    # near, far
    perspective: bool
    projection_gl: np.ndarray = field(repr=False, default=None)

    @property
    def view_projection(self) -> np.ndarray:
        return (self.projection @ self.view).astype(np.float32)


@dataclass
class Light:
    name: str
    kind: str                    # SUN, POINT, SPOT or AREA
    color: np.ndarray
    energy: float
    position: np.ndarray
    direction: np.ndarray        # the way the light points


@dataclass
class SceneData:
    positions: np.ndarray        # (vertices, 3) world space; every three rows are one triangle
    normals: np.ndarray
    uvs: np.ndarray
    colors: np.ndarray           # the base colour of each triangle's material
    objects: list                # (name, first vertex, vertex count)
    camera: Camera
    lights: list
    resolution: tuple[int, int]

    def __repr__(self):
        return (f"SceneData({len(self.positions) // 3} triangles in {len(self.objects)} objects, "
                f"camera {self.camera.name!r}, {len(self.lights)} lights, {self.resolution[0]}x{self.resolution[1]})")


def open(path: str):
    """Load a .blend file as the current scene."""
    _bpy().ops.wm.open_mainfile(filepath=os.fspath(path))


def _resolution(scene, resolution):
    if resolution:
        return int(resolution[0]), int(resolution[1])
    r = scene.render
    return round(r.resolution_x * r.resolution_percentage / 100), round(r.resolution_y * r.resolution_percentage / 100)


def _camera(scene, deps, camera, size) -> Camera:
    bpy = _bpy()
    cam = bpy.data.objects[camera] if isinstance(camera, str) else (camera or scene.camera)
    if cam is None or cam.type != "CAMERA":
        raise ValueError("the scene has no active camera: add one, or pass camera=<object or name>")
    world = cam.matrix_world.normalized()       # ignore any scale on the camera object
    gl = np.array(cam.calc_matrix_camera(deps, x=size[0], y=size[1],
                                         scale_x=scene.render.pixel_aspect_x, scale_y=scene.render.pixel_aspect_y), np.float64)
    return Camera(cam.name, np.array(world.inverted(), np.float64), GL_TO_WEBGPU @ gl, np.array(world.translation, np.float32),
                  (cam.data.clip_start, cam.data.clip_end), cam.data.type == "PERSP", gl)


def _material_color(material):
    if material is None:
        return DEFAULT_COLOR
    if material.use_nodes and material.node_tree:
        for node in material.node_tree.nodes:
            if node.type == "BSDF_PRINCIPLED":
                return tuple(node.inputs["Base Color"].default_value[:3])
    return tuple(material.diffuse_color[:3])


def scene(resolution=None, camera=None) -> SceneData:
    """The current Blender scene as arrays: every visible mesh, with modifiers applied, in world space."""
    bpy = _bpy()
    sc = bpy.context.scene
    deps = bpy.context.evaluated_depsgraph_get()
    size = _resolution(sc, resolution)
    positions, normals, uvs, colors, objects, lights, first = [], [], [], [], [], [], 0

    for inst in deps.object_instances:
        obj = inst.object
        world = np.array(inst.matrix_world, np.float64)
        if obj.type == "LIGHT":
            rot = world[:3, :3] / np.linalg.norm(world[:3, :3], axis=0)
            lights.append(Light(obj.name, obj.data.type, np.array(obj.data.color, np.float32), float(obj.data.energy),
                                world[:3, 3].astype(np.float32), (-rot[:, 2]).astype(np.float32)))
            continue
        if obj.type not in ("MESH", "CURVE", "SURFACE", "FONT", "META") or not inst.show_self:
            continue
        mesh = obj.to_mesh()
        if mesh is None or not len(mesh.polygons):
            continue
        mesh.calc_loop_triangles()
        tris = len(mesh.loop_triangles)
        loops = np.empty(tris * 3, np.int32)
        mesh.loop_triangles.foreach_get("loops", loops)
        corner_vertex = np.empty(len(mesh.loops), np.int32)
        mesh.loops.foreach_get("vertex_index", corner_vertex)
        co = np.empty(len(mesh.vertices) * 3, np.float32)
        mesh.vertices.foreach_get("co", co)
        corner_normal = np.empty(len(mesh.loops) * 3, np.float32)
        mesh.corner_normals.foreach_get("vector", corner_normal)
        material_index = np.empty(tris, np.int32)
        mesh.loop_triangles.foreach_get("material_index", material_index)

        p = co.reshape(-1, 3)[corner_vertex[loops]].astype(np.float64)
        n = corner_normal.reshape(-1, 3)[loops].astype(np.float64)
        p = p @ world[:3, :3].T + world[:3, 3]
        n = n @ np.linalg.inv(world[:3, :3])            # normals transform by the inverse transpose
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-20)
        if mesh.uv_layers.active:
            uv = np.empty(len(mesh.loops) * 2, np.float32)
            mesh.uv_layers.active.data.foreach_get("uv", uv)
            uv = uv.reshape(-1, 2)[loops]
        else:
            uv = np.zeros((tris * 3, 2), np.float32)
        palette = np.array([_material_color(s.material) for s in obj.material_slots] or [DEFAULT_COLOR], np.float32)
        c = np.repeat(palette[np.clip(material_index, 0, len(palette) - 1)], 3, axis=0)
        if np.linalg.det(world[:3, :3]) < 0:            # a mirrored object: restore counter-clockwise winding
            order = np.arange(tris * 3).reshape(-1, 3)[:, ::-1].reshape(-1)
            p, n, uv, c = p[order], n[order], uv[order], c[order]
        positions.append(p.astype(np.float32)); normals.append(n.astype(np.float32)); uvs.append(uv); colors.append(c)
        objects.append((obj.name, first, tris * 3))
        first += tris * 3
        obj.to_mesh_clear()

    if not positions:
        raise ValueError("the scene has no visible mesh objects")
    return SceneData(np.concatenate(positions), np.concatenate(normals), np.concatenate(uvs), np.concatenate(colors),
                     objects, _camera(sc, deps, camera, size), lights, size)


def reference(resolution=None, camera=None) -> dict:
    """What a correct renderer must agree with, computed by Blender's ray casting.

    Returns arrays with one entry per pixel, row 0 at the top:
      hit     True where the pixel's centre ray meets a surface
      depth   distance of that surface in front of the camera, along the view axis (inf if none)
      ndc     the value a WebGPU depth buffer holds for that surface (1 if none)
      normal  the surface's geometric normal in world space
    """
    bpy = _bpy()
    from mathutils import Vector

    sc = bpy.context.scene
    deps = bpy.context.evaluated_depsgraph_get()
    w, h = _resolution(sc, resolution)
    cam = _camera(sc, deps, camera, (w, h))
    unproject = np.linalg.inv(cam.projection_gl @ cam.view)
    xs = (np.arange(w) + 0.5) / w * 2 - 1
    ys = 1 - (np.arange(h) + 0.5) / h * 2

    def world(z):
        ndc = np.stack(np.broadcast_arrays(xs[None, :], ys[:, None], z, 1.0), -1)
        p = ndc @ unproject.T
        return p[..., :3] / p[..., 3:4]

    near, far = world(-1.0), world(1.0)
    direction = far - near
    length = np.linalg.norm(direction, axis=-1, keepdims=True)
    direction /= length
    hit = np.zeros((h, w), bool)
    depth = np.full((h, w), np.inf, np.float32)
    ndc_depth = np.ones((h, w), np.float32)
    normal = np.zeros((h, w, 3), np.float32)
    view_projection = cam.projection @ cam.view
    for y in range(h):
        for x in range(w):
            ok, location, n, _index, _obj, _matrix = sc.ray_cast(deps, Vector(near[y, x]), Vector(direction[y, x]),
                                                                  distance=float(length[y, x, 0]))
            if ok:
                point = np.array([*location, 1.0])
                clip = view_projection @ point
                hit[y, x] = True
                depth[y, x] = -(cam.view @ point)[2]
                ndc_depth[y, x] = clip[2] / clip[3]
                normal[y, x] = n
    return {"hit": hit, "depth": depth, "ndc": ndc_depth, "normal": normal, "camera": cam, "resolution": (w, h)}


def render(resolution=None, camera=None, samples: int = 32, transparent: bool = False) -> np.ndarray:
    """Blender's own picture of the scene, rendered with Cycles on the CPU: (height, width, 4) uint8."""
    bpy = _bpy()
    from PIL import Image

    sc = bpy.context.scene
    w, h = _resolution(sc, resolution)
    r = sc.render
    saved = (r.engine, r.resolution_x, r.resolution_y, r.resolution_percentage, r.filepath, r.film_transparent, sc.camera)
    try:
        r.engine = "CYCLES"
        sc.cycles.device = "CPU"
        sc.cycles.samples = samples
        r.resolution_x, r.resolution_y, r.resolution_percentage = w, h, 100
        r.film_transparent = transparent
        r.image_settings.file_format = "PNG"
        r.image_settings.color_mode = "RGBA"
        if camera is not None:
            sc.camera = bpy.data.objects[camera] if isinstance(camera, str) else camera
        with tempfile.TemporaryDirectory() as tmp:
            r.filepath = os.path.join(tmp, "render.png")
            bpy.ops.render.render(write_still=True)
            return np.asarray(Image.open(r.filepath).convert("RGBA")).copy()
    finally:
        r.engine, r.resolution_x, r.resolution_y, r.resolution_percentage, r.filepath, r.film_transparent, sc.camera = saved


def demo_scene():
    """Replace the current scene with a small one made from Blender's built-in shapes:
    the monkey head on a floor, a camera and a sun. Handy when no .blend file is at hand."""
    bpy = _bpy()
    bpy.ops.wm.read_factory_settings(use_empty=True)

    def material(name, color):
        m = bpy.data.materials.new(name)
        m.diffuse_color = (*color, 1.0)                 # what the viewport shows
        if m.node_tree is None:
            m.use_nodes = True
        for node in m.node_tree.nodes:                  # what the renderers use
            if node.type == "BSDF_PRINCIPLED":
                node.inputs["Base Color"].default_value = (*color, 1.0)
        return m

    bpy.ops.mesh.primitive_monkey_add(location=(0, 0, 0.55), rotation=(0.15, 0, 0.5))
    monkey = bpy.context.active_object
    bpy.ops.object.shade_smooth()
    monkey.data.materials.append(material("clay", (0.80, 0.42, 0.25)))
    bpy.ops.mesh.primitive_plane_add(size=6, location=(0, 0, -0.45))
    bpy.context.active_object.data.materials.append(material("floor", (0.35, 0.45, 0.55)))
    bpy.ops.mesh.primitive_cube_add(size=0.7, location=(-1.5, 0.6, -0.1), rotation=(0, 0, 0.6))
    bpy.context.active_object.data.materials.append(material("box", (0.25, 0.6, 0.35)))

    bpy.ops.object.camera_add(location=(3.2, -4.0, 2.2))
    camera = bpy.context.active_object
    from mathutils import Vector

    camera.rotation_euler = (Vector((0, 0, 0.2)) - camera.location).to_track_quat("-Z", "Y").to_euler()   # aim at the monkey
    bpy.context.scene.camera = camera
    bpy.ops.object.light_add(type="SUN", location=(4, -2, 6), rotation=(0.65, 0.15, 0.9))
    bpy.context.active_object.data.energy = 3.0
    # Show colours as they are, without a film-like tone curve, so pictures are easy to compare.
    bpy.context.scene.view_settings.view_transform = "Standard"
    return bpy.context.scene
