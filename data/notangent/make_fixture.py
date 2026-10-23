#!/usr/bin/env python3
"""Generates data/notangent: the box scene without TANGENT, plus an occluder.

The box's primitive loses its TANGENT attribute, so the loader generates its
tangents with MikkTSpace. A second node, offset beside the box, holds a
primitive with no material, no TANGENT and no TEXCOORD_0, so the loader gives
it the fallback tangent basis. The saver writes both sets into the output.

Run from anywhere: python3 data/notangent/make_fixture.py
"""

import json
import pathlib
import shutil

HERE = pathlib.Path(__file__).resolve().parent
BOX = HERE.parent / "box"


def main():
    gltf = json.loads((BOX / "scene.gltf").read_text())
    shutil.copyfile(BOX / "scene.bin", HERE / "scene.bin")

    box = gltf["meshes"][0]["primitives"][0]
    del box["attributes"]["TANGENT"]

    occluder = {
        "name": "Occluder",
        "primitives": [{
            "attributes": {"POSITION": box["attributes"]["POSITION"],
                           "NORMAL": box["attributes"]["NORMAL"]},
            "indices": box["indices"],
        }],
    }
    gltf["meshes"].append(occluder)
    gltf["nodes"].append({"mesh": len(gltf["meshes"]) - 1, "name": "Occluder",
                          "translation": [1.5, 0.0, 0.0],
                          "scale": [0.5, 0.5, 0.5]})
    gltf["scenes"][0]["nodes"].append(len(gltf["nodes"]) - 1)

    (HERE / "scene.gltf").write_text(json.dumps(gltf, indent=2) + "\n")


if __name__ == "__main__":
    main()
