from slepc4py import SLEPc

import fenics as fe

import logging
import numpy as np

from scipy.linalg import cho_factor, cho_solve
from scipy.sparse.linalg import aslinearoperator, eigsh
from scipy.spatial.distance import pdist, squareform
from scipy.special import gamma, kv

from .utils import build_observation_operator

from pykeops.numpy import LazyTensor

logger = logging.getLogger(__name__)
fe.set_log_level(40)


def sq_exp_covariance(grid, scale, ell):
    """
    Squared exponential covariance function.

    Parameters
    ----------
    grid : ndarray
        Spatial grid of shape (n_points, n_dimensions).
    scale : float
        Variance hyperparameter.
    ell : float
        Length-scale hyperparameter.
    """
    dist = pdist(grid, metric="euclidean")
    dist = squareform(dist)

    K = scale**2 * np.exp(-dist**2 / (2 * ell**2))
    K[np.diag_indices_from(K)] += 1e-10
    return K


def sq_exp_spectral_density(omega, scale, ell, D=1):
    """
    Squared exponential spectral density.

    Parameters
    ----------
    omega : float
        Frequency at which to evaluate the spectral density.
    scale : float
        Variance hyperparameter.
    ell : float
        Length-scale hyperparameter.
    """
    return (scale**2 * (2 * np.pi * ell**2)**(D / 2) *
            np.exp(-omega**2 * ell**2 / 2))


def sq_exp_evd(grid, scale, ell, k=32):
    K_dense = sq_exp_covariance(grid=grid, scale=scale, ell=ell)
    vals, vecs = eigsh(K_dense, k=k)
    return vals, vecs


def sq_exp_evd_keops(grid, scale, ell, k=32):
    """
    Approximate a squared-exponential covariance matrix using KeOps.

    If a GPU device is detected then it will use this to accelerate
    computations.

    Parameters
    ----------
    grid : ndarray
        Spatial grid of shape (n_points, n_dimensions).
    scale : float
        Variance hyperparameter.
    ell : float
        Length-scale hyperparameter.
    k : int
        Number of desired eigenvalues.
    """
    from pykeops.config import gpu_available
    logger.info("Approximating GP with KeOps, using GPU: %s",
                str(gpu_available))

    # compute in single, return in double
    dtype_compute = "float32"
    dtype_return = "float64"

    def sq_exp_keops(grid, scale, ell):
        x_ = grid / ell
        x_i, x_j = LazyTensor(x_[:, None, :]), LazyTensor(x_[None, :, :])
        K = (-((x_i - x_j)**2).sum(2) / 2).exp()
        K *= scale**2
        return K

    grid = grid.astype(dtype_compute)
    K_keops = sq_exp_keops(grid, scale, ell)
    assert K_keops.dtype == "float32"

    K_keops = aslinearoperator(K_keops)
    vals, vecs = eigsh(K_keops, k=k)

    return vals.astype(dtype_return), vecs.astype(dtype_return)


def laplacian_evd(comm, V, k=64, bc="Dirichlet", return_function=False):
    """
    Aproximate the smallest k eigenvalues/eigenfunctions of the Laplacian.

    I.e. solve u_xx + u_yy + u_zz + ... = - lambda u,
    using shift-invert mode in SLEPc for scalable computations.

    Parameters
    ----------
    comm: mpi4py.MPI.COMM
        MPI communicator on which we do the computing. Usually MPI.COMM_SELF or
        MPI.COMM_WORLD.
    V : fenics.FunctionSpace
        FunctionSpace on which to compute the approximation.
    k : int, optional
        Number of modes to take in the approximation.
    bc : str, optional
        Boundary conditions to use in the approximation. Either 'Dirichlet' or
        'Neumann'.
    return_function: bool, optional
        Flag to return a list of fenics functions instead of a numpy.ndarray.
        This allows for interpolation across different FunctionSpaces.

    Returns
    -------
    numpy.ndarray
        Eigenvalues of the Laplacian, computed using the FunctionSpace V.
    numpy.ndarray : optional
        Vector of nodal values of the eigenfunctions, computed on the
        FunctionSpace V.
    list : optional
        List of eigenfunctions, defined on the FunctionSpace V.
    """
    def boundary(x, on_boundary):
        return on_boundary

    bc_types = ["Dirichlet", "Neumann"]
    if bc not in bc_types:
        raise ValueError("Invalid bc, expected one of {bc_types}")
    elif bc == "Dirichlet":
        bc = fe.DirichletBC(V, fe.Constant(0), boundary)
    else:
        bc = None

    # define variational problem
    u = fe.TrialFunction(V)
    v = fe.TestFunction(V)

    a = fe.inner(fe.grad(u), fe.grad(v)) * fe.dx
    A = fe.PETScMatrix(comm)
    fe.assemble(a, tensor=A)

    M = fe.PETScMatrix(comm)
    M_no_bc = fe.PETScMatrix(comm)
    fe.assemble(fe.inner(u, v) * fe.dx, tensor=M)
    fe.assemble(fe.inner(u, v) * fe.dx, tensor=M_no_bc)

    if bc is not None:
        # sets BC rows of A to identity
        bc.apply(A)

        # sets rows of M to zeros
        bc.apply(M)
        bc.zero(M)

    M = M.mat()
    M_no_bc = M_no_bc.mat()
    A = A.mat()

    # solver inspired by: cmaurini
    # https://gist.github.com/cmaurini/6dea21fc01c6a07caeb96ff9c86dc81e
    E = SLEPc.EPS(comm)
    E.create()
    E.setOperators(A, M)
    E.setDimensions(nev=k, ncv=2*k)
    E.setWhichEigenpairs(E.Which.TARGET_MAGNITUDE)
    E.setTarget(0)
    E.setTolerances(1e-12, 100_000)
    S = E.getST()
    S.setType("sinvert")
    E.setFromOptions()
    E.solve()

    # check that things have converged
    logger.info("Eigenvalues converged: %d", E.getConverged())

    # and set up objects for storage
    vr, wr = A.getVecs()
    vi, wi = A.getVecs()

    laplace_eigenvals = np.zeros((k, ))
    errors = np.zeros((k, ))

    if return_function:
        eigenfunctions = [fe.Function(V) for i in range(k)]
    else:
        eigenvecs = np.zeros((vr.array_r.shape[0], k))

    for i in range(k):
        laplace_eigenvals[i] = np.real(E.getEigenpair(i, vr, vi))
        errors[i] = E.computeError(i)

        # normalize by weighted norm on  the function space
        M_no_bc.mult(vr, wr)
        ef_norm = np.sqrt(wr.tDot(vr))

        # scale by norm on the function space
        if return_function:
            eigenfunctions[i].vector().set_local(vr.getArray() / ef_norm)
        else:
            eigenvecs[:, i] = vr.array_r / ef_norm

    # what are we returning
    if return_function:
        return (laplace_eigenvals, eigenfunctions)
    else:
        return (laplace_eigenvals, eigenvecs)


def sq_exp_evd_hilbert(comm, V, k=64, scale=1., ell=1., bc="Dirichlet"):
    """
    Approximate the SqExp covariance using Hilbert-GP.

    For full details:
    Solin, A., Särkkä, S., 2020. Hilbert space methods for reduced-rank
    Gaussian process regression. Stat Comput 30, 419–446.
    https://doi.org/10.1007/s11222-019-09886-w

    Parameters
    ----------
    V : fenics.FunctionSpace
        FunctionSpace on which to compute the approximation.
    k : int, optional
        Number of modes to take in the approximation.
    scale : float
        Variance hyperparameter.
    ell : float
        Length-scale hyperparameter.
    bc : str, optional
        Boundary conditions to use in the approximation. Either 'Dirichlet' or
        'Neumann'.
    """
    laplace_eigenvals, eigenvecs = laplacian_evd(
        comm, V, k=k, bc=bc, return_function=False)

    # enforce positivity --- picks up eigenvalues that are negative
    laplace_eigenvals = laplace_eigenvals
    logger.info("Laplacian eigenvalues: %s", laplace_eigenvals)
    eigenvals = sq_exp_spectral_density(
        np.sqrt(laplace_eigenvals),
        scale=scale,
        ell=ell,
        D=V.mesh().geometric_dimension())
    return (eigenvals, eigenvecs)


def matern_covariance(grid, scale=1., ell=1., nu=2):
    """
    Compute Matern covariance matrix.

    Parameters
    ----------
    grid : ndarray
        Spatial grid of shape (n_points, n_dimensions).
    scale : float
        Variance hyperparameter.
    ell : float
        Length-scale hyperparameter.
    nu : float
        Smoothness parameter.
    """
    kappa = np.sqrt(2 * nu) / ell
    dist = pdist(grid, metric="euclidean")
    dist = squareform(dist)
    dist[dist == 0.0] += np.finfo(float).eps  # strict zeros result in nan

    K = (scale**2 / (2**(nu - 1) * gamma(nu))
         * (kappa * dist)**nu * kv(nu, kappa * dist))
    K[np.diag_indices_from(K)] += 1e-8
    return K


def matern_spectral_density(omega, scale=1., ell=1., nu=2):
    """
    Matern spectral density.

    Parameters
    ----------
    omega : float
        Frequency at which to evaluate the spectral density.
    scale : float
        Variance hyperparameter.
    ell : float
        Length-scale hyperparameter.
    nu : float
        Smoothness parameter.
    """
    kappa = np.sqrt(2 * nu) / ell
    return (scale**2 * np.sqrt(4 * np.pi) * gamma(nu + 1 / 2)
            * kappa**(2 * nu) / gamma(nu)
            * (kappa**2 + 4 * np.pi**2 * omega**2)**(-(nu + 1 / 2)))


# TODO(connor): include support for nonzero means
# TODO(connor): include support for matern spectral representation
class SqExpHilbertGP:
    def __init__(self, V, k, bcs="Dirichlet"):
        self.V = V
        self.k = k
        self.D = V.mesh().geometric_dimension()

        # compute eigendecomposition
        self.eigenvals, self.phi = laplacian_evd(V=V, k=k, bc=bcs)
        self.spectral_density = sq_exp_spectral_density

        # initialise parameters
        self.sigma = np.exp(np.random.normal())
        self.rho, self.ell = np.random.uniform(size=(2, ))

    def set_dataset(self, x, y):
        self.x = x
        self.y = y
        self.n_obs = len(y)
        self.H = build_observation_operator(self.x, self.V)

    def set_priors(self, rho, ell, sigma):
        """ Set prior functions. """
        self.rho_prior = rho
        self.ell_prior = ell
        self.sigma_prior = sigma

    def lml(self):
        """ Log marginal likelihood. """
        S = self.spectral_density(
            np.sqrt(self.eigenvals), self.rho, self.ell, self.D)
        Z = (self.H @ self.phi).T @ (self.H @ self.phi)
        Z[np.diag_indices_from(Z)] += self.sigma / S
        Z_chol = cho_factor(Z, lower=True)

        phi_trans_y = (self.H @ self.phi).T @ self.y
        Z_chol_inv_phi_trans_y = cho_solve(Z_chol, phi_trans_y)

        log_det = (2 * (self.n_obs - self.k) * np.log(self.sigma)
                   + 2 * np.sum(np.log(np.diag(Z_chol[0])))
                   + np.sum(np.log(S)))
        maha = np.dot(self.y, self.y)
        maha -= Z_chol_inv_phi_trans_y.T @ Z_chol_inv_phi_trans_y
        maha *= 1 / self.sigma**2
        return (0.5 * log_det + 0.5 * maha
                + 0.5 * self.n_obs * np.log(2 * np.pi))

    def grad_lml(self):
        """ Gradient of log marginal likelihood. """
        pass

    def lmp(self):
        """ Log marginal posterior. """
        return self.lml()
