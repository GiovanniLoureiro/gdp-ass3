import unittest
import numpy as np
import bmesh
import mathutils
from .deform import gradient_deform, constrained_gradient_deform
from assignment3.matrices.util import numpy_verts


class TestDeformation(unittest.TestCase):

    def setUp(self):
        self.mesh = self.create_test_mesh()

    def create_test_mesh(self):
        bm = bmesh.new()
        verts = [bm.verts.new(co) for co in [(0, 0, 0), (1, 0, 0), (0, 1, 0)]]
        bm.faces.new(verts)
        bm.verts.ensure_lookup_table()
        bm.faces.ensure_lookup_table()

        for i, vert in enumerate(bm.verts):
            vert.index = i
        for i, face in enumerate(bm.faces):
            face.index = i

        return bm

    def test_gradient_deform(self):
        A = mathutils.Matrix(((2, 0, 0), (0, 2, 0), (0, 0, 2)))  # Scaling matrix
        original_verts = numpy_verts(self.mesh)
        new_verts = gradient_deform(self.mesh, A)

        # Check if new vertices have been modified correctly
        self.assertEqual(new_verts.shape, original_verts.shape)
        self.assertFalse(np.allclose(original_verts, new_verts))

    def test_constrained_gradient_deform(self):
        A = mathutils.Matrix(((2, 0, 0), (0, 2, 0), (0, 0, 2)))  # Scaling matrix
        original_verts = numpy_verts(self.mesh)
        selected_faces = [0]  # Only one face in this simple test mesh
        new_verts = constrained_gradient_deform(self.mesh, selected_faces, A)

        # Check if new vertices have been modified correctly
        self.assertEqual(new_verts.shape, original_verts.shape)
        self.assertFalse(np.allclose(original_verts, new_verts))

    def test_barycenter_constraint(self):
        A = mathutils.Matrix(((2, 0, 0), (0, 2, 0), (0, 0, 2)))  # Scaling matrix
        original_verts = numpy_verts(self.mesh)
        original_barycenter = np.mean(original_verts, axis=0)

        new_verts = gradient_deform(self.mesh, A)
        new_barycenter = np.mean(new_verts, axis=0)

        # Ensure the barycenter remains constant
        np.testing.assert_almost_equal(original_barycenter, new_barycenter, decimal=5)


if __name__ == '__main__':
    unittest.main()
