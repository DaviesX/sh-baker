#!/usr/bin/env python3
"""Generates data/layers: the box scene with exporter-shaped SH_material_layers.

The box keeps its geometry and sun. Its material becomes a two-layer stack
covering every tcMod kind and both WAVE rgbGen forms (with and without
`func`). A second, smaller box beside it carries a one-layer additive stack
(blendSrc/blendDst ONE, surfaceBlend ADD) with an animMap whose frame 0 is
the layer's texture, plus KHR_materials_emissive_strength, so the bake turns
it into an area light that reaches the red box's lightmap.

Every non-index number is a float-exact decimal literal (30.0, not 30), as
ioq3-map-exporter writes them; texture indices and baseLayer are integers.

Run from anywhere: python3 data/layers/make_fixture.py
"""

import json
import pathlib
import shutil
import struct
import zlib

HERE = pathlib.Path(__file__).resolve().parent
BOX = HERE.parent / "box"


def write_png(path, width, height, rgba):
    """Writes an 8-bit RGBA PNG. `rgba` is a row-major list of (r, g, b, a)."""
    raw = b"".join(
        b"\x00" + bytes(c for px in rgba[y * width:(y + 1) * width] for c in px)
        for y in range(height))

    def chunk(tag, data):
        body = tag + data
        return (struct.pack(">I", len(data)) + body +
                struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF))

    png = (b"\x89PNG\r\n\x1a\n" +
           chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)) +
           chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))
    path.write_bytes(png)


def checker(a, b):
    """A 4x4 two-colour checkerboard."""
    return [a if (x + y) % 2 == 0 else b for y in range(4) for x in range(4)]


def main():
    gltf = json.loads((BOX / "scene.gltf").read_text())
    shutil.copyfile(BOX / "scene.bin", HERE / "scene.bin")

    write_png(HERE / "layer0.png", 4, 4,
              checker((200, 180, 160, 255), (120, 100, 80, 255)))
    write_png(HERE / "layer1.png", 4, 4,
              checker((255, 255, 255, 255), (160, 160, 160, 192)))
    write_png(HERE / "flame0.png", 4, 4,
              checker((255, 160, 40, 255), (255, 96, 16, 255)))

    gltf["images"] = [{"uri": "layer0.png"}, {"uri": "layer1.png"},
                      {"uri": "flame0.png"}]
    gltf["textures"] = [{"source": 0}, {"source": 1}, {"source": 2}]

    layered = {
        "name": "Layered",
        "pbrMetallicRoughness": {
            "baseColorFactor": [0.800000011920929, 0.0, 0.0, 1.0],
            "metallicFactor": 0.0,
        },
        "extensions": {
            "SH_material_layers": {
                "surfaceBlend": "BLEND",
                "cullMode": "NONE",
                "baseLayer": 1,
                "layers": [
                    {
                        "texture": {"index": 0},
                        "blendSrc": "ONE",
                        "blendDst": "ZERO",
                        "rgbGen": {"type": "WAVE", "func": "SIN", "base": 0.5,
                                   "amplitude": 0.25, "phase": 0.0,
                                   "frequency": 1.0},
                        "tcMod": [
                            {"type": "SCALE", "value": [2.0, 3.0]},
                            {"type": "SCROLL", "value": [0.5, 0.0]},
                            {"type": "ROTATE", "value": 30.0},
                            {"type": "TURB",
                             "value": ["SIN", 0.0, 0.125, 0.0, 1.0]},
                            {"type": "STRETCH",
                             "value": ["NONE", 1.0, 0.5, 0.0, 2.0]},
                            {"type": "TRANSFORM",
                             "value": [1.0, 0.0, 0.5, 0.0, 1.0, 0.0]},
                        ],
                    },
                    {
                        # WAVE without func: the exporter writes this for noise
                        # and unknown wave functions.
                        "texture": {"index": 1},
                        "blendSrc": "DST_COLOR",
                        "blendDst": "ZERO",
                        "rgbGen": {"type": "WAVE", "base": 0.75,
                                   "amplitude": 0.25, "phase": 0.25,
                                   "frequency": 0.5},
                    },
                ],
            }
        },
    }
    flame = {
        "name": "Flame",
        "pbrMetallicRoughness": {
            "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
            "metallicFactor": 0.0,
        },
        "extensions": {
            "KHR_materials_emissive_strength": {"emissiveStrength": 20.0},
            "SH_material_layers": {
                "surfaceBlend": "ADD",
                "cullMode": "FRONT",
                "baseLayer": 0,
                "layers": [
                    {
                        # animMap: frame 0 is the layer's own texture.
                        "texture": {"index": 2},
                        "animFreq": 5.0,
                        "animFrames": [2, 1],
                        "blendSrc": "ONE",
                        "blendDst": "ONE",
                        "rgbGen": {"type": "IDENTITY"},
                    },
                ],
            },
        },
    }
    gltf["materials"] = [layered, flame]

    flame_mesh = json.loads(json.dumps(gltf["meshes"][0]))
    flame_mesh["name"] = "FlameMesh"
    flame_mesh["primitives"][0]["material"] = 1
    gltf["meshes"].append(flame_mesh)
    gltf["nodes"].append({"mesh": len(gltf["meshes"]) - 1, "name": "Flame",
                          "translation": [1.5, 0.0, 0.0],
                          "scale": [0.5, 0.5, 0.5]})
    gltf["scenes"][0]["nodes"].append(len(gltf["nodes"]) - 1)

    for ext in ("SH_material_layers", "KHR_materials_emissive_strength"):
        if ext not in gltf["extensionsUsed"]:
            gltf["extensionsUsed"].append(ext)

    (HERE / "scene.gltf").write_text(json.dumps(gltf, indent=2) + "\n")


if __name__ == "__main__":
    main()
