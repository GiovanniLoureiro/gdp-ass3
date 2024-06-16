import numpy as np
from scipy.sparse.linalg import lsqr
from scipy.sparse import eye as sparse_eye
from scipy.sparse import csr_matrix
import bmesh
import mathutils
from assignment3.matrices.util import numpy_verts
from assignment3.matrices.differential_coordinates import build_gradient_matrix, build_mass_matrices

def gradient_deform(mesh: bmesh.types.BMesh, A: mathutils.Matrix) -> np.ndarray:
    # Get current vertex positions as a numpy array
    verts = numpy_verts(mesh)

    # Convert A to a NumPy array
    A_np = np.array(A)

    # Build the gradient matrix G and mass matrices M, Mv
    G = build_gradient_matrix(mesh)
    M, Mv = build_mass_matrices(mesh)

    # Transform the gradients
    num_faces = len(mesh.faces)
    Gx = G @ verts[:, 0]
    Gy = G @ verts[:, 1]
    Gz = G @ verts[:, 2]

    transformed_Gx = A_np @ np.stack((Gx[:num_faces], Gx[num_faces:2*num_faces], Gx[2*num_faces:]), axis=0)
    transformed_Gy = A_np @ np.stack((Gy[:num_faces], Gy[num_faces:2*num_faces], Gy[2*num_faces:]), axis=0)
    transformed_Gz = A_np @ np.stack((Gz[:num_faces], Gz[num_faces:2*num_faces], Gz[2*num_faces:]), axis=0)

    target_Gx = np.hstack(transformed_Gx.T)
    target_Gy = np.hstack(transformed_Gy.T)
    target_Gz = np.hstack(transformed_Gz.T)

    # Solve for new vertex positions using least squares
    new_x = lsqr(G, target_Gx)[0]
    new_y = lsqr(G, target_Gy)[0]
    new_z = lsqr(G, target_Gz)[0]

    # Combine the new coordinates into a single array
    new_verts = np.vstack((new_x, new_y, new_z)).T

    # Keep the barycenter constant
    original_barycenter = np.mean(verts, axis=0)
    new_barycenter = np.mean(new_verts, axis=0)
    translation = original_barycenter - new_barycenter
    new_verts += translation

    return new_verts

def constrained_gradient_deform(
        mesh: bmesh.types.BMesh,
        selected_face_indices: list[int],
        A: mathutils.Matrix
) -> np.ndarray:
    # Get current vertex positions as a numpy array
    verts = numpy_verts(mesh)

    # Convert A to a NumPy array
    A_np = np.array(A)

    # Build the gradient matrix G and mass matrices M, Mv
    G = build_gradient_matrix(mesh)
    M, Mv = build_mass_matrices(mesh)

    # Transform the gradients for the selected faces
    num_faces = len(mesh.faces)
    Gx = G @ verts[:, 0]
    Gy = G @ verts[:, 1]
    Gz = G @ verts[:, 2]

    selected_Gx = np.zeros_like(Gx)
    selected_Gy = np.zeros_like(Gy)
    selected_Gz = np.zeros_like(Gz)

    for face_idx in selected_face_indices:
        selected_Gx[3 * face_idx:3 * face_idx + 3] = A_np @ Gx[3 * face_idx:3 * face_idx + 3]
        selected_Gy[3 * face_idx:3 * face_idx + 3] = A_np @ Gy[3 * face_idx:3 * face_idx + 3]
        selected_Gz[3 * face_idx:3 * face_idx + 3] = A_np @ Gz[3 * face_idx:3 * face_idx + 3]

    # Solve for new vertex positions using least squares
    new_x = lsqr(G, selected_Gx)[0]
    new_y = lsqr(G, selected_Gy)[0]
    new_z = lsqr(G, selected_Gz)[0]

    # Combine the new coordinates into a single array
    new_verts = np.vstack((new_x, new_y, new_z)).T

    # Keep the barycenter constant
    original_barycenter = np.mean(verts, axis=0)
    new_barycenter = np.mean(new_verts, axis=0)
    translation = original_barycenter - new_barycenter
    new_verts += translation

    return new_verts
