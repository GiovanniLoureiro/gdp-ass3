import unittest
import numpy as np
from mathutils import Vector
import bmesh
from .differential_coordinates import *
from scipy.sparse import coo_matrix, coo_array


class TestGradient(unittest.TestCase):

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

    def test_triangle_gradient(self):
        face = self.mesh.faces[0]
        gradient = triangle_gradient(face)
        self.assertEqual(gradient.shape, (3, 3))
        # Additional checks for gradient values can be added here

    def test_build_gradient_matrix(self):
        G = build_gradient_matrix(self.mesh)
        num_faces, num_verts = len(self.mesh.faces), len(self.mesh.verts)
        self.assertEqual(G.shape, (3 * num_faces, num_verts))
        # Ensure G is a sparse matrix
        self.assertIsInstance(G, coo_array)
        # Additional checks for specific entries in G can be added here


class TestMassMatrices(unittest.TestCase):

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

    def test_build_mass_matrices(self):
        M, Mv = build_mass_matrices(self.mesh)
        num_faces, num_verts = len(self.mesh.faces), len(self.mesh.verts)
        self.assertEqual(M.shape, (num_verts, num_verts))
        self.assertEqual(Mv.shape, (3 * num_faces, 3 * num_faces))
        # Ensure M and Mv are sparse arrays
        self.assertIsInstance(M, coo_array)
        self.assertIsInstance(Mv, coo_array)
        # Additional checks for specific entries in M and Mv can be added here


class TestCotangentMatrix(unittest.TestCase):

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

    def test_build_cotangent_matrix(self):
        G = build_gradient_matrix(self.mesh)
        _, Mv = build_mass_matrices(self.mesh)
        S = build_cotangent_matrix(G, Mv)
        num_faces, num_verts = len(self.mesh.faces), len(self.mesh.verts)
        self.assertEqual(S.shape, (num_verts, num_verts))
        self.assertTrue((S - S.T).nnz == 0)


if __name__ == '__main__':
    unittest.main()
