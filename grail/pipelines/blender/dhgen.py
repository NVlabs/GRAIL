"""Import DHGen characters without losing their native rig or baked materials.

The source USD stays unchanged. OmniSurface's baked color, roughness, normals
and cutout maps are translated to Principled; this is not an MDL renderer.
"""

import glob
import json
import math
from pathlib import Path

import bpy
from mathutils import Matrix, Vector
from pxr import Usd, UsdGeom, UsdShade


def open_dhgen_stage(path):
    stage = Usd.Stage.Open(str(path))
    if not stage or not stage.GetPrimAtPath("/DHGen_Root"):
        return None
    if UsdGeom.GetStageUpAxis(stage) != "Y" or not math.isclose(
        UsdGeom.GetStageMetersPerUnit(stage), 0.01
    ):
        raise ValueError("DHGen importer requires the reviewed Y-up centimeter USD contract")
    return stage


def _value(attribute, visited=None):
    """Resolve authored constants or DHGen's baked file-texture graph outputs."""
    if not attribute:
        return None
    visited = set() if visited is None else visited
    key = str(attribute.GetPath())
    if key in visited:
        raise ValueError(f"Cyclic DHGen shader input: {key}")
    visited.add(key)
    connections = attribute.GetConnections()
    if connections:
        if len(connections) != 1:
            raise ValueError(f"Unsupported multi-source DHGen shader input: {key}")
        return _value(attribute.GetStage().GetAttributeAtPath(connections[0]), visited)
    result = attribute.Get()
    if result is not None:
        return result
    if attribute.GetName().startswith("inputs:"):
        return None  # Unauthored MDL parameters use their shader defaults.
    if attribute.GetName().startswith("outputs:"):
        prim = attribute.GetPrim()
        texture = prim.GetAttribute("inputs:texture")
        if texture:
            return _value(texture, visited)
        for name in ("inputs:a", "inputs:input"):
            if prim.GetAttribute(name):
                return _value(prim.GetAttribute(name), visited)
    raise ValueError(f"Unsupported unresolved DHGen shader input: {key}")


def _image(path, color):
    matches = sorted(glob.glob(str(path).replace("<UDIM>", "[0-9][0-9][0-9][0-9]")))
    if not matches:
        raise FileNotFoundError(f"Missing DHGen texture: {path}")
    image = bpy.data.images.load(matches[0], check_existing=False)
    if "<UDIM>" in str(path):
        image.source = "TILED"
        image.filepath = str(path)
        for filename in matches:
            number = int(Path(filename).stem.rsplit("_", 1)[-1])
            if number not in {tile.number for tile in image.tiles}:
                image.tiles.new(number)
        image.reload()
    image.colorspace_settings.name = "sRGB" if color else "Non-Color"
    return image


def _material(stage, path, directory):
    prim = stage.GetPrimAtPath(path)
    shaders = [p for p in Usd.PrimRange(prim) if p.IsA(UsdShade.Shader)]
    base = next((p for p in shaders if p.GetName() == "OmniSurfaceBase"), None)
    if base is None:
        base = next((p for p in shaders if p.GetName() == "Shader"), None)
    if base is None:
        raise ValueError(f"Unsupported DHGen material: {path}")
    kind = base.GetAttribute("info:mdl:sourceAsset:subIdentifier").Get()
    if kind not in ("OmniSurfaceBase", "OmniSurface_ClearWater", "OmniPBR"):
        raise ValueError(f"Unsupported DHGen shader {kind}: {path}")
    material = bpy.data.materials.new("DHGen_" + prim.GetName())
    material.use_nodes = True
    nodes, links = material.node_tree.nodes, material.node_tree.links
    nodes.clear()
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    output = nodes.new("ShaderNodeOutputMaterial")
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    texture_paths = []

    def channel(name, socket, default=None, color=False, normal=False):
        value = _value(base.GetAttribute("inputs:" + name))
        if value is None:
            value = default
        if value is None:
            return
        if hasattr(value, "path"):
            if not value.path:
                return
            image_path = (directory / value.path).resolve()
            tex = nodes.new("ShaderNodeTexImage")
            tex.image = _image(image_path, color)
            texture_paths.append(str(image_path))
            source = tex.outputs["Color"]
            if normal:
                converter = nodes.new("ShaderNodeNormalMap")
                links.new(source, converter.inputs["Color"])
                source = converter.outputs["Normal"]
            links.new(source, bsdf.inputs[socket])
        elif not normal:
            bsdf.inputs[socket].default_value = (*value, 1.0) if color else float(value)

    if kind == "OmniSurface_ClearWater":
        bsdf.inputs["Base Color"].default_value = (1, 1, 1, 1)
        bsdf.inputs["Transmission Weight"].default_value = 1
        bsdf.inputs["IOR"].default_value = 1.333
        channel("specular_reflection_roughness", "Roughness", 0.08)
    elif kind == "OmniPBR":
        # The supplied DHGen hidden-face material explicitly masks body faces
        # beneath clothing. Do not expose those faces as a neutral material.
        opacity = _value(base.GetAttribute("inputs:opacity_constant"))
        if prim.GetName() != "HIDDEN_mtl" or opacity != 0:
            raise ValueError(f"Only the explicit DHGen hidden OmniPBR is supported: {path}")
        bsdf.inputs["Alpha"].default_value = 0
    else:
        channel("diffuse_reflection_color", "Base Color", (0.8, 0.8, 0.8), color=True)
        channel("specular_reflection_roughness", "Roughness", 0.5)
        channel("metalness", "Metallic", 0)
        channel("specular_reflection_ior", "IOR", 1.5)
        channel("specular_reflection_weight", "Specular IOR Level", 0.5)
        channel("coat_weight", "Coat Weight", 0)
        channel("coat_roughness", "Coat Roughness", 0.1)
        channel("coat_color", "Coat Tint", (1, 1, 1), color=True)
        channel("geometry_normal", "Normal", normal=True)
        if _value(base.GetAttribute("inputs:enable_specular_transmission")):
            channel("specular_transmission_weight", "Transmission Weight", 0)
        if _value(base.GetAttribute("inputs:enable_opacity")):
            channel("geometry_opacity", "Alpha", 1)
            threshold = _value(base.GetAttribute("inputs:geometry_opacity_threshold"))
            if threshold and bsdf.inputs["Alpha"].is_linked:
                source = bsdf.inputs["Alpha"].links[0].from_socket
                cutoff = nodes.new("ShaderNodeMath")
                cutoff.operation = "GREATER_THAN"
                cutoff.inputs[1].default_value = threshold
                links.new(source, cutoff.inputs[0])
                links.new(cutoff.outputs[0], bsdf.inputs["Alpha"])
    material.surface_render_method = "DITHERED"
    material["source_material"] = str(path)
    material["source_textures"] = json.dumps(texture_paths)
    material["conversion"] = (
        "DHGen baked maps to Principled; MDL scattering/retroreflection approximate"
    )
    return material


def prepare_import(stage, imported_objects, parent, source):
    """Preserve hierarchy, restore source Y-up coordinates and bind materials."""
    correction = Matrix.Rotation(-math.pi / 2, 4, "X")
    for obj in imported_objects:
        if obj.parent not in imported_objects:
            original = obj.matrix_world.copy()
            obj.parent = parent
            obj.matrix_world = correction @ original
    parent["character_rig"] = "dhgen"
    parent["source_meters_per_unit"] = 0.01
    material_cache = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        found = [o for o in imported_objects if o.type == "MESH" and o.name == prim.GetName()]
        if not found:
            continue  # Invisible USD meshes remain excluded by the importer.
        if len(found) != 1:
            raise ValueError(f"Ambiguous imported DHGen mesh: {prim.GetPath()}")
        obj = found[0]
        obj.data.materials.clear()

        def assign(bound, faces, obj):
            material, _ = UsdShade.MaterialBindingAPI(bound).ComputeBoundMaterial()
            if not material:
                return
            path = str(material.GetPath())
            if path not in material_cache:
                material_cache[path] = _material(stage, path, Path(source).parent)
            mat = material_cache[path]
            index = obj.data.materials.find(mat.name)
            if index < 0:
                index = len(obj.data.materials)
                obj.data.materials.append(mat)
            for i in faces:
                obj.data.polygons[i].material_index = index

        assign(prim, range(len(obj.data.polygons)), obj)
        for subset in UsdGeom.Subset.GetGeomSubsets(UsdGeom.Imageable(prim)):
            if subset.GetElementTypeAttr().Get() == "face":
                assign(subset.GetPrim(), subset.GetIndicesAttr().Get(), obj)
        if not obj.data.materials:
            raise ValueError(f"DHGen mesh has no bound material: {prim.GetPath()}")
    bpy.context.view_layer.update()


def repose(parent):
    """Lower arms in anatomical source space, independent of USD bone axes."""
    armatures = [obj for obj in parent.children_recursive if obj.type == "ARMATURE"]
    if len(armatures) != 1:
        raise ValueError("Expected one native DHGen armature")
    armature = armatures[0]
    for side, sign in (("Left", 1), ("Right", -1)):
        for joint, child, target in (
            ("Arm", "ForeArm", (sign * 0.25, -0.965, 0.08)),
            ("ForeArm", "Hand", (sign * 0.10, -0.95, 0.30)),
        ):
            bone = armature.pose.bones[side + joint]
            endpoint = armature.pose.bones[side + child]
            current = endpoint.head - bone.head
            delta = current.rotation_difference(Vector(target).normalized()).to_matrix().to_4x4()
            pivot = Matrix.Translation(bone.head)
            location, scale = bone.location.copy(), bone.scale.copy()
            bone.rotation_mode = "QUATERNION"
            bone.matrix = pivot @ delta @ pivot.inverted() @ bone.matrix
            bone.location, bone.scale = location, scale
            bpy.context.view_layer.update()
    parent["initial_pose"] = "dhgen_anatomical_relaxed_arms"


def mask_material(original, color):
    """Keep authored hair cutouts and hidden-face opacity in the SAM2 seed."""
    material = original.copy()
    material.name = original.name + "_mask"
    nodes, links = material.node_tree.nodes, material.node_tree.links
    bsdf = next(node for node in nodes if node.type == "BSDF_PRINCIPLED")
    output = next(node for node in nodes if node.type == "OUTPUT_MATERIAL")
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = color
    transparent = nodes.new("ShaderNodeBsdfTransparent")
    mix = nodes.new("ShaderNodeMixShader")
    links.new(transparent.outputs[0], mix.inputs[1])
    links.new(emission.outputs[0], mix.inputs[2])
    alpha = bsdf.inputs["Alpha"]
    if alpha.is_linked:
        links.new(alpha.links[0].from_socket, mix.inputs[0])
    else:
        mix.inputs[0].default_value = alpha.default_value
    links.new(mix.outputs[0], output.inputs["Surface"])
    return material
