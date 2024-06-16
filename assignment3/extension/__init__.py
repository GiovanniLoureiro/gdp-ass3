import bpy
import bmesh
import numpy as np
from scipy.sparse import eye, csc_matrix
from scipy.sparse.linalg import spsolve
from assignment3.matrices.differential_coordinates import build_gradient_matrix, build_mass_matrices, \
    build_cotangent_matrix
from assignment3.matrices.util import numpy_verts


def laplace_smooth(mesh: bmesh.types.BMesh, lambda_factor: float = 0.1) -> np.ndarray:
    verts = numpy_verts(mesh)
    G = build_gradient_matrix(mesh)
    M, Mv = build_mass_matrices(mesh)
    S = build_cotangent_matrix(G, Mv)

    num_verts = verts.shape[0]
    I = eye(num_verts)
    L = csc_matrix(S)

    A = I - lambda_factor * L
    new_verts_x = spsolve(A, verts[:, 0])
    new_verts_y = spsolve(A, verts[:, 1])
    new_verts_z = spsolve(A, verts[:, 2])

    new_verts = np.vstack((new_verts_x, new_verts_y, new_verts_z)).T

    return new_verts


class MESH_OT_LaplaceSmooth(bpy.types.Operator):
    bl_idname = "mesh.laplace_smooth"
    bl_label = "Laplace Smooth"
    bl_options = {'REGISTER', 'UNDO'}

    lambda_factor: bpy.props.FloatProperty(
        name="Lambda Factor",
        description="Smoothing factor",
        default=0.1,
        min=0.0,
        max=1.0
    )

    def execute(self, context):
        obj = context.object
        if obj.type != 'MESH':
            self.report({'ERROR'}, "Active object is not a mesh")
            return {'CANCELLED'}

        mesh = bmesh.new()
        mesh.from_mesh(obj.data)
        new_verts = laplace_smooth(mesh, self.lambda_factor)

        for vert, new_vert in zip(mesh.verts, new_verts):
            vert.co = new_vert

        mesh.to_mesh(obj.data)
        mesh.free()

        return {'FINISHED'}


class MESH_PT_LaplaceSmoothPanel(bpy.types.Panel):
    bl_label = "Laplace Smooth"
    bl_idname = "MESH_PT_laplace_smooth"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'TOOLS'
    bl_category = 'Tools'

    def draw(self, context):
        layout = self.layout
        layout.operator("mesh.laplace_smooth")


def register():
    bpy.utils.register_class(MESH_OT_LaplaceSmooth)
    bpy.utils.register_class(MESH_PT_LaplaceSmoothPanel)


def unregister():
    bpy.utils.unregister_class(MESH_OT_LaplaceSmooth)
    bpy.utils.unregister_class(MESH_PT_LaplaceSmoothPanel)


if __name__ == "__main__":
    register()
