import numpy as np


def compute_vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute per-vertex normals of a triangular mesh."""
    verts = np.asarray(verts, dtype=float)
    faces = np.asarray(faces, dtype=int)
    normals = np.zeros_like(verts)
    # compute face normals
    tris = verts[faces]
    face_normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    # normalize face normals
    lengths = np.linalg.norm(face_normals, axis=1)
    face_normals /= (lengths[:, None] + 1e-8)
    # accumulate normals per vertex
    for i, f in enumerate(faces):
        for vi in f:
            normals[vi] += face_normals[i]
    # normalize vertex normals
    v_lengths = np.linalg.norm(normals, axis=1)
    normals /= (v_lengths[:, None] + 1e-8)
    return normals


def generate_thick_shell(verts: np.ndarray, faces: np.ndarray, thickness: float):
    """
    Create a shell mesh with nacreous thickness by offsetting in and out along normals.
    Returns (new_verts, new_faces).
    """
    verts = np.asarray(verts, dtype=float)
    faces = np.asarray(faces, dtype=int)
    normals = compute_vertex_normals(verts, faces)
    # split thickness equally for inner and outer offsets
    half = thickness / 2.0
    outer = verts + normals * half
    inner = verts - normals * half
    # outer and inner faces
    outer_faces = faces.copy()
    # reverse inner faces to correct orientation
    inner_faces = faces[:, ::-1] + len(outer)
    # find boundary edges for side walls
    edge_count = {}
    for tri in faces:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            key = tuple(sorted((a, b)))
            edge_count[key] = edge_count.get(key, 0) + 1
    boundary = [edge for edge, cnt in edge_count.items() if cnt == 1]
    side_faces = []
    for a, b in boundary:
        # connect outer to inner
        side_faces.append((a, b, b + len(outer)))
        side_faces.append((a, b + len(outer), a + len(outer)))
    # assemble new mesh
    new_verts = np.vstack((outer, inner))
    new_faces = np.vstack((outer_faces, inner_faces, np.array(side_faces, dtype=int)))
    return new_verts, new_faces


def compress_mesh(verts: np.ndarray, axis: str = 'z', factor: float = 0.5) -> np.ndarray:
    """Compress the mesh along a specified axis by a given factor."""
    idx = {'x': 0, 'y': 1, 'z': 2}[axis.lower()]
    new = np.array(verts, dtype=float)
    new[:, idx] *= factor
    return new


def shear_mesh(verts: np.ndarray, axis: str = 'x', shear_axis: str = 'y', shear_factor: float = 0.2) -> np.ndarray:
    """Shear the mesh: new_axis += shear_factor * shear_axis."""
    new = np.array(verts, dtype=float)
    i = {'x': 0, 'y': 1, 'z': 2}[axis.lower()]
    j = {'x': 0, 'y': 1, 'z': 2}[shear_axis.lower()]
    new[:, i] += shear_factor * new[:, j]
    return new


def flatten_mesh(verts: np.ndarray, axis: str = 'z', flatten_factor: float = 0.5) -> np.ndarray:
    """Flatten the mesh along an axis (alias for compression)."""
    return compress_mesh(verts, axis, flatten_factor)


def add_nacre_layer(verts: np.ndarray, faces: np.ndarray, thickness: float = 0.05):
    """
    Extrude the mesh along its normals to simulate a nacreous material layer.
    Returns new vertices of the nacre layer.
    """
    normals = compute_vertex_normals(verts, faces)
    return verts + normals * thickness


def apply_flatten(verts: np.ndarray, axis: str = 'z', factor: float = 0.5) -> np.ndarray:
    """
    Compress the mesh along a given axis by a factor (<1 for flattening).
    axis: one of 'x','y','z'
    factor: scale multiplier along that axis
    """
    out = verts.copy()
    idx = {'x':0, 'y':1, 'z':2}.get(axis.lower(), 2)
    out[:, idx] *= factor
    return out


def apply_shear(verts: np.ndarray, shear_axis: str = 'x', direction_axis: str = 'y', factor: float = 0.3) -> np.ndarray:
    """
    Apply a simple shear: move shear_axis coordinate by factor * direction_axis coordinate.
    shear_axis: axis to shear ('x','y','z')
    direction_axis: axis that drives the shear
    factor: shear factor
    """
    sa = {'x':0, 'y':1, 'z':2}.get(shear_axis.lower(), 0)
    da = {'x':0, 'y':1, 'z':2}.get(direction_axis.lower(), 1)
    out = verts.copy()
    out[:, sa] += verts[:, da] * factor
    return out


def apply_taphonomy(verts: np.ndarray,
                    faces: np.ndarray,
                    flatten: dict = None,
                    shear: dict = None,
                    nacre_thickness: float = 0.0) -> dict:
    """
    Apply a sequence of taphonomic distortions and optionally add nacreous layer.
    flatten: {'axis':str, 'factor':float}
    shear: {'shear_axis':str, 'direction_axis':str, 'factor':float}
    nacre_thickness: positive float to extrude a nacre layer

    Returns a dict with keys 'verts': distorted verts,
    and if nacre_thickness>0, 'nacre_verts': extruded layer verts.
    """
    res_verts = verts.copy()
    if flatten:
        res_verts = apply_flatten(res_verts, axis=flatten.get('axis','z'), factor=flatten.get('factor',1.0))
    if shear:
        res_verts = apply_shear(res_verts,
                                 shear_axis=shear.get('shear_axis','x'),
                                 direction_axis=shear.get('direction_axis','y'),
                                 factor=shear.get('factor',0.0))
    result = {'verts': res_verts, 'faces': faces}
    if nacre_thickness and nacre_thickness > 0:
        result['nacre_verts'] = add_nacre_layer(res_verts, faces, thickness=nacre_thickness)
    return result


def restore_taphonomy(verts: np.ndarray,
                      faces: np.ndarray,
                      flatten: dict = None,
                      shear: dict = None) -> np.ndarray:
    """
    Restore original geometry by inverting taphonomic distortions.
    flatten: {'axis':str, 'factor':float}
    shear: {'shear_axis':str, 'direction_axis':str, 'factor':float}

    Returns restored vertex positions.
    """
    out = verts.copy()
    # invert shear first
    if shear:
        sa = {'x':0,'y':1,'z':2}.get(shear.get('shear_axis','x').lower(),0)
        da = {'x':0,'y':1,'z':2}.get(shear.get('direction_axis','y').lower(),1)
        out[:, sa] -= shear.get('factor',0.0) * out[:, da]
    # invert flatten
    if flatten:
        idx = {'x':0,'y':1,'z':2}.get(flatten.get('axis','z').lower(),2)
        factor = flatten.get('factor',1.0)
        if factor != 0:
            out[:, idx] /= factor
    return out
