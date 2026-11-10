"""Blender as the scene and camera: export a scene, rasterize it with a technique, and
check the result against what Blender's own ray casting says each pixel should see.

Needs Blender's Python module:  pip install bpy
    PYTHONPATH=src python3 tests/test_blender.py
"""
import numpy as np

import colabgraphics as cg
from colabgraphics import blender

SHADER = """
struct VSOut
{
    float4 position : SV_Position;
    float3 world    : TEXCOORD0;
    float3 normal   : NORMAL;
    float3 color    : COLOR0;
};

struct PSOut
{
    float4 color : SV_Target0;
    float  depth : SV_Target1;
};

VSOut VSMain(Vertex v)
{
    VSOut o;
    o.position = mul(vars.view_projection, float4(v.position, 1));
    o.world = v.position;
    o.normal = v.normal;
    o.color = v.color;
    return o;
}

PSOut PSMain(VSOut input)
{
    float lambert = max(dot(normalize(input.normal), -vars.sun_direction), 0.0);
    PSOut o;
    o.color = float4(input.color * (0.15 + 0.85 * lambert), 1);
    o.depth = -mul(vars.view, float4(input.world, 1)).z;   // distance in front of the camera
    return o;
}
"""


def build(scene_data):
    t = cg.Technique("blender_scene")
    w, h = scene_data.resolution
    color = t.texture("color", size=(w, h))
    depth = t.texture("view_depth", size=(w, h), format="r32float")
    mesh = t.mesh("scene", scene_data.positions, scene_data.normals, scene_data.uvs, scene_data.colors)
    t.var("view_projection", scene_data.camera.view_projection)
    t.var("view", scene_data.camera.view.astype(np.float32))
    t.var("sun_direction", tuple(scene_data.lights[0].direction))
    t.draw("raster", SHADER, targets=[color, depth], mesh=mesh, depth=True, cull="back")
    return t, color, depth


def test_rasterized_scene_agrees_with_blender_ray_casting():
    blender.demo_scene()
    data = blender.scene(resolution=(256, 192))
    assert len(data.objects) == 3 and len(data.lights) == 1, data
    t, color, depth = build(data)
    image, ours = t.read([color, depth])
    ours = ours[..., 0]
    ref = blender.reference(resolution=(256, 192))

    covered = ours > 0
    union = (covered | ref["hit"]).sum()
    iou = (covered & ref["hit"]).sum() / union
    assert iou > 0.995, f"coverage differs from Blender: IoU {iou:.4f}"

    # Compare depth away from silhouettes, where one pixel can legitimately see a different surface.
    both = covered & ref["hit"]
    inner = both.copy()
    for axis in (0, 1):
        for shift in (-1, 1):
            inner &= np.roll(both, shift, axis)
    d = np.where(ref["hit"], ref["depth"], 0.0)
    grad = np.abs(np.diff(d, axis=1, prepend=d[:, :1])) + np.abs(np.diff(d, axis=0, prepend=d[:1]))
    smooth = inner & (grad < 0.05)
    error = np.abs(ours - ref["depth"])[smooth]
    assert smooth.sum() > 0.5 * both.sum()
    assert np.median(error) < 1e-4 and np.percentile(error, 99) < 5e-3, (np.median(error), np.percentile(error, 99))
    assert image.shape == (192, 256, 4) and image[..., :3].max() > 100
    return {"iou": float(iou), "depth_median_error": float(np.median(error)), "depth_p99_error": float(np.percentile(error, 99)),
            "pixels_compared": int(smooth.sum())}


if __name__ == "__main__":
    print("ok   test_rasterized_scene_agrees_with_blender_ray_casting", test_rasterized_scene_agrees_with_blender_ray_casting())
