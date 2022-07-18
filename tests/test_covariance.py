import pytest

import numpy as np
import matplotlib.pyplot as plt

import fenics as fe
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh

from statfenics.covariance import (sq_exp_covariance, sq_exp_spectral_density,
                                   sq_exp_evd, sq_exp_evd_keops,
                                   sq_exp_evd_hilbert, matern_covariance)


def boundary(x, on_boundary):
    return on_boundary


mesh = fe.UnitIntervalMesh(200)
V = fe.FunctionSpace(mesh, "CG", 1)
x_grid = V.tabulate_dof_coordinates()
scale = 1.
ell = 1e-1
k = 128


def test_sq_exp():
    scale = 1.
    ell = 1e-1

    # closure over scale, ell, to test G
    def test_G(x_grid):
        G = sq_exp_covariance(x_grid, scale=scale, ell=ell)
        np.testing.assert_allclose(G[0, 0], 1.)
        assert G.shape[0] == x_grid.shape[0]

    # verify on 1D
    mesh = fe.UnitIntervalMesh(200)
    V = fe.FunctionSpace(mesh, "CG", 1)
    x_grid = V.tabulate_dof_coordinates()
    test_G(x_grid)

    # now verify on 2D
    mesh = fe.UnitSquareMesh(8, 8)
    V = fe.FunctionSpace(mesh, "CG", 1)
    x_grid = V.tabulate_dof_coordinates()
    test_G(x_grid)


def test_sq_exp_evd():
    rtol = 1e-4  # small rtol due to single precision
    norm = np.linalg.norm

    # check approx equal eigenvalues
    k_test = 32
    G = sq_exp_covariance(x_grid, scale, ell)
    G_vals, G_vecs = sq_exp_evd(x_grid, scale, ell, k=k_test)
    G_approx = G_vecs @ np.diag(G_vals) @ G_vecs.T
    rel_error = norm(G - G_approx) / norm(G)
    print(rel_error)
    assert rel_error <= rtol


def test_sq_exp_evd_keops():
    rtol = 1e-4  # small rtol due to single precision
    norm = np.linalg.norm

    k_test = 16
    G = sq_exp_covariance(x_grid, scale, ell)
    G_vals, G_vecs = sq_exp_evd(x_grid, scale, ell, k=k_test)
    G_vals_keops, G_vecs_keops = sq_exp_evd_keops(x_grid,
                                                  scale=scale,
                                                  ell=ell,
                                                  k=k_test)

    # check approx equal eigenvalues
    np.testing.assert_allclose(G_vals[-k_test:], G_vals_keops, rtol=rtol)
    G_approx = G_vecs_keops @ np.diag(G_vals_keops) @ G_vecs_keops.T
    rel_error = norm(G - G_approx) / norm(G)
    assert rel_error <= rtol


def test_sq_exp_evd_hilbert():
    vals, vecs = sq_exp_evd_hilbert(V, k=k, scale=scale, ell=ell)

    # first two should be deleted for better approx.
    assert vals.shape == (128, )
    assert vecs.shape == (x_grid.shape[0], 128)

    # check ordering
    vals_sorted = np.sort(vals)[::-1]
    np.testing.assert_almost_equal(vals, vals_sorted)


def test_sq_exp_evd_hilbert_2d():
    mesh = fe.UnitSquareMesh(32, 32)
    V = fe.FunctionSpace(mesh, "CG", 1)
    scale, ell = 1., 1e-1
    k = 32

    bc = fe.DirichletBC(V, fe.Constant(0), boundary)

    # mass matrix used to ensure orthogonality on the weighted inner product
    # <u, v> = u M v'
    u, v = fe.TrialFunction(V), fe.TestFunction(V)
    M = fe.PETScMatrix()
    fe.assemble(u * v * fe.dx, tensor=M)
    bc.apply(M)
    M = M.mat()
    M_scipy = csr_matrix(M.getValuesCSR()[::-1], shape=M.size)

    vals, vecs = sq_exp_evd_hilbert(V, k=k, scale=scale, ell=ell)

    assert vals.shape == (32, )
    assert vecs.shape == (1089, 32)

    # check orthogonality wrt mass matrix
    np.testing.assert_almost_equal(vecs[:, 0] @ M_scipy @ vecs[:, 0], 1.)
    np.testing.assert_almost_equal(vecs[:, 0] @ M_scipy @ vecs[:, 1], 0.)


def test_sq_exp_evd_hilbert_neumann():
    mesh = fe.UnitSquareMesh(32, 32)
    V = fe.FunctionSpace(mesh, "CG", 1)
    scale, ell = 1., 1e-1

    # check orthogonality on the weighted inner product
    # <u, v> = u M v'
    u, v = fe.TrialFunction(V), fe.TestFunction(V)
    M = fe.PETScMatrix()
    fe.assemble(u * v * fe.dx, tensor=M)
    M = M.mat()
    M_scipy = csr_matrix(M.getValuesCSR()[::-1], shape=M.size)

    vals, vecs = sq_exp_evd_hilbert(V,
                                    k=32,
                                    scale=scale,
                                    ell=ell,
                                    bc="Neumann")

    assert vals.shape == (32, )
    assert vecs.shape == (1089, 32)

    # check orthogonality wrt mass matrix
    np.testing.assert_almost_equal(vecs[:, 0] @ M_scipy @ vecs[:, 0], 1.)
    np.testing.assert_almost_equal(vecs[:, 0] @ M_scipy @ vecs[:, 1], 0.)

    analytical_eigenvalues = np.array(
        [0, np.pi**2, np.pi**2, 2 * np.pi**2, 4 * np.pi**2, 4 * np.pi**2])
    spectral_density = sq_exp_spectral_density(np.sqrt(analytical_eigenvalues),
                                               scale,
                                               ell,
                                               D=2)
    np.testing.assert_array_almost_equal(vals[:6], spectral_density, decimal=4)


def test_matern():
    scale = 1.
    ell = 1e-1

    def test_G_matern(x_grid):
        # check that things are the same for nu = 0.5
        dist = pdist(x_grid, metric="euclidean")
        dist = squareform(dist)
        G_simple = scale**2 * np.exp(-dist / ell)
        G = matern_covariance(x_grid, scale=scale, ell=ell, nu=0.5)
        np.testing.assert_allclose(G, G_simple)

        # check again for nu = 2.5
        G_simple = (scale**2
                    * (1 + np.sqrt(5) * dist / ell + 5 * dist**2 / (3 * ell**2))
                    * np.exp(-np.sqrt(5) * dist / ell))
        G = matern_covariance(x_grid, scale=scale, ell=ell, nu=5 / 2)
        np.testing.assert_allclose(G, G_simple)

    # verify first in 1D
    mesh = fe.UnitIntervalMesh(200)
    V = fe.FunctionSpace(mesh, "CG", 1)
    x_grid = V.tabulate_dof_coordinates()
    test_G_matern(x_grid)

    # and now in 2D
    mesh = fe.UnitSquareMesh(8, 8)
    V = fe.FunctionSpace(mesh, "CG", 1)
    x_grid = V.tabulate_dof_coordinates()
    test_G_matern(x_grid)
