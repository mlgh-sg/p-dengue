import numpy as np
import sympy as sp


def bspline_basis_full_knots_numeric(full_knots, degree, x, simplify=False):
    """
    Returns list of basis functions for all degrees up to `degree`. Works with symbolic or numeric knots.

    Output:
        basis[d][i] = i-th basis function of degree d
    """

    # Full knots include knots outside the domain of interest, be they equispaced or repeated at the boundaries.
    # Number of knots is n = num_interior_knots + 2*(degree+1).
    # These will define n - (degree+1) = num_interior_knots + degree + 1 basis functions of degree d.
    n = len(full_knots)

    # Store basis functions by degree
    basis = []

    # --- Degree 0 ---
    B0 = []
    for i in range(n-1):
        if full_knots[i] == full_knots[i+1]:
            B0.append(sp.Integer(0))
        else:
            if i == n-2:
                # boundary adjustment for the last basis function to ensure it is 1 at the last knot
                B0.append(
                    sp.Piecewise(
                        (1, (x >= full_knots[i]) & (x <= full_knots[i+1])),
                        (0, True)))
            else:
                B0.append(
                    sp.Piecewise(
                        (1, (x >= full_knots[i]) & (x < full_knots[i+1])),
                        (0, True)))
    basis.append(B0)

    # --- Higher degrees ---
    for d in range(1, degree + 1):
        Bd = []
        prev = basis[d - 1]

        for i in range(len(prev) - 1):
            # w terms in recursion formula
            denom1 = full_knots[i + d] - full_knots[i]
            denom2 = full_knots[i + d + 1] - full_knots[i + 1]

            term1 = 0
            term2 = 0
            # conditions to handle repeated knots and avoid division by zero
            if denom1 != 0:
                term1 = ((x - full_knots[i]) / denom1) * prev[i]
            if denom2 != 0:
                term2 = ((full_knots[i + d + 1] - x) / denom2) * prev[i + 1]

            Bd.append(sp.simplify(term1 + term2) if simplify else term1 + term2)

        basis.append(Bd)
    return basis

def generate_spline_basis_full_knots_numeric(degree, full_knots):
    x = sp.Symbol('x', real=True)
    basis = bspline_basis_full_knots_numeric(full_knots, degree, x)

    return {
        "x": x,
        "full_knots": full_knots,
        "interior_knots": full_knots[degree+1:-(degree+1)],
        "basis": basis
    }

def generate_spline_basis_equispaced_numeric(degree, t_0, t_n, num_knots):
    # Domain boundaries, define inter knot distance
    h = (t_n - t_0) / (num_knots + 1)
    # Full knots include knots outside the domain boundaries
    full_knots = np.linspace(t_0 - degree * h, t_n + degree * h, num_knots + 2*degree + 2)

    return generate_spline_basis_full_knots_numeric(degree, full_knots)

def eval_spline_basis_equispaced_numeric(degree, t_0, t_n, num_knots, x_vals):
    # Evaluate the basis functions at the given x values. This constructs the B spline design matrix.
    B = generate_spline_basis_equispaced_numeric(degree, t_0, t_n, num_knots)
    x = B["x"]
    full_knots = B["full_knots"]
    interior_knots = B["interior_knots"]
    basis = B["basis"]

    B_vals = np.zeros((len(x_vals), len(basis[degree])))
    for i in range(len(basis[degree])):
        B_func = sp.lambdify(x, basis[degree][i], "numpy")
        B_vals[:, i] = B_func(x_vals)
    return {"B": B_vals, "x": x_vals, "full_knots": full_knots, "interior_knots": interior_knots}