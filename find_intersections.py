import numpy as np
import cupy as cp
from math import ceil, floor

from utils import tocpu, togpu

# Define the coefficients for the two polynomials
coeflist = [
    [1.26581062e+08, -6.65625087e+06, 8.42679006e+04],
    [1.26581062e+08, -9.20875231e+06, 1.64245710e+05]
]


# Define the start position and the range around it
tstart2 = 0.030318928843199852
epsilon = 1e-6

# Function to find crossing points where poly(x) = (2n +1)*pi
def find_crossings(coef, x_min, x_max):
    y_min = cp.polyval(coef, x_min)
    y_max = cp.polyval(coef, x_max)
    y_lower = min(y_min, y_max)
    y_upper = max(y_min, y_max)

    # Compute the range of n values
    n_min = ceil((y_lower - cp.pi) / (2 * cp.pi))
    n_max = floor((y_upper - cp.pi) / (2 * cp.pi))

    crossings = []
    for n in range(int(n_min), int(n_max) + 1):
        # Solve poly(x) - (2n +1)*pi = 0
        shifted_poly = coef.copy()
        shifted_poly[2] -= (2 * n + 1) * cp.pi
        roots = togpu(np.roots(tocpu(shifted_poly)))
        # Filter real roots within [x_min, x_max]
        real_roots = roots[cp.isreal(roots)].real
        valid_roots = real_roots[(real_roots >= x_min) & (real_roots <= x_max)]
        crossings.extend(valid_roots.tolist())
    return crossings

# Function to determine n for wrapping within a section
def determine_n(coef, x_start, x_end):
    # Choose the midpoint to determine n
    x_mid = (x_start + x_end) / 2
    y_mid = cp.polyval(coef, x_mid)
    n = floor((y_mid + cp.pi) / (2 * cp.pi))
    return n


def find_intersections(coefa, coefb, tstart2, epsilon):
    x_min = tstart2 - epsilon
    x_max = tstart2 + epsilon

    # Find all crossing points for both polynomials
    crossings_poly1 = find_crossings(coefa, x_min, x_max)
    crossings_poly2 = find_crossings(coefb, x_min, x_max)

    # Combine and sort all crossing points
    all_crossings = sorted(crossings_poly1 + crossings_poly2)
    # Ensure the boundaries are included
    all_crossings = [x_min] + all_crossings + [x_max]
    # Remove duplicate crossings (if any) within a tolerance
    unique_crossings = []
    tol = 1e-12
    for x in all_crossings:
        if not unique_crossings or abs(x - unique_crossings[-1]) > tol:
            unique_crossings.append(x)
    all_crossings = unique_crossings

    # Define sections between consecutive crossings
    sections = []
    for i in range(len(all_crossings) - 1):
        sections.append((all_crossings[i], all_crossings[i + 1]))



    # Initialize list to store intersection points
    intersection_points = []

    # Initialize Plotly figure with subplots

    # Determine number of sections to create subplots

    # Iterate over each section to find intersections and plot
    for idx, (x_start, x_end) in enumerate(sections):
        # Determine n for each polynomial
        n1 = determine_n(coefa, x_start, x_end)
        n2 = determine_n(coefb, x_start, x_end)

        # Adjust the constant term for wrapping
        # Create shifted polynomials by adding (2n -1)*pi to the constant term
        poly1_shifted_coef = coefa.copy()
        poly1_shifted_coef -= (2 * n1) * cp.pi

        poly2_shifted_coef = coefb.copy()
        poly2_shifted_coef -= (2 * n2) * cp.pi

        # Compute the difference polynomial
        poly_diff = cp.polysub(poly1_shifted_coef, poly2_shifted_coef)

        # Find roots of the difference polynomial
        roots = togpu(np.roots(tocpu(poly_diff)))
        # Filter real roots
        real_roots = roots[cp.isreal(roots)].real
        # Filter roots within the current section
        valid_roots = real_roots[(real_roots >= x_start) & (real_roots <= x_end)]
        # Convert to float and remove duplicates
        valid_roots = valid_roots.astype(float)

        # Add valid roots to intersection points
        intersection_points.extend(valid_roots)

    return intersection_points, poly1_shifted_coef, poly2_shifted_coef
