import numpy as np
from scipy.sparse import coo_matrix, coo_array, csr_matrix
import bmesh

def triangle_gradient(triangle: bmesh.types.BMFace) -> np.ndarray:
    assert len(triangle.verts) == 3
    verts = triangle.verts
    v0, v1, v2 = np.array(verts[0].co), np.array(verts[1].co), np.array(verts[2].co)

    e1 = v1 - v0
    e2 = v2 - v0
    normal = np.cross(e1, e2)
    area = np.linalg.norm(normal) / 2.0
    normal = normal / np.linalg.norm(normal)

    e0 = v2 - v1
    e1 = v0 - v2
    e2 = v1 - v0

    local_gradient = np.vstack((np.cross(normal, e0),
                                np.cross(normal, e1),
                                np.cross(normal, e2))).T / (2.0 * area)

    return local_gradient

def build_gradient_matrix(mesh: bmesh.types.BMesh) -> coo_array:
    num_faces, num_verts = len(mesh.faces), len(mesh.verts)

    row_indices = []
    col_indices = []
    data = []

    for face in mesh.faces:
        local_grad = triangle_gradient(face)
        for i, vert in enumerate(face.verts):
            row_indices.extend([3 * face.index + j for j in range(3)])
            col_indices.extend([vert.index] * 3)
            data.extend(local_grad[:, i])

    return coo_array((data, (row_indices, col_indices)), shape=(3 * num_faces, num_verts))

def build_mass_matrices(mesh: bmesh.types.BMesh) -> tuple[coo_array, coo_array]:
    num_faces, num_verts = len(mesh.faces), len(mesh.verts)

    M_data = np.zeros(num_verts)
    Mv_data = np.zeros(3 * num_faces)

    for face in mesh.faces:
        verts = face.verts
        v0, v1, v2 = np.array(verts[0].co), np.array(verts[1].co), np.array(verts[2].co)
        area = np.linalg.norm(np.cross(v1 - v0, v2 - v0)) / 2.0
        for vert in verts:
            M_data[vert.index] += area
        Mv_data[3 * face.index: 3 * face.index + 3] = area

    M = coo_array((M_data, (np.arange(num_verts), np.arange(num_verts))), shape=(num_verts, num_verts))
    Mv = coo_array((Mv_data, (np.arange(3 * num_faces), np.arange(3 * num_faces))),
                   shape=(3 * num_faces, 3 * num_faces))

    return M, Mv

def build_cotangent_matrix(G: coo_array, Mv: coo_array) -> coo_array:
    S = csr_matrix(G.T @ Mv @ G)
    return S.tocoo()
