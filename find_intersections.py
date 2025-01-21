import numpy as np
import cupy as cp
import plotly.graph_objects as go
from math import ceil, floor
from pltfig import pltfig

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
def find_crossings(coef_in, x_min, x_max):
    coef = coef_in.copy()
    # coef[2] %= (2 * cp.pi)
    # ymin ymax: suppose is increasing/decreasing in this section
    assert not x_min <= - coef[1] / 2 / coef[0] <= x_max

    ya = cp.polyval(coef, x_min)
    yb = cp.polyval(coef, x_max)
    y_lower = min(ya, yb)
    y_upper = max(ya, yb)

    # Compute the range of n values
    n_min = ceil((y_lower - cp.pi) / (2 * cp.pi))
    n_max = floor((y_upper - cp.pi) / (2 * cp.pi))

    crossings = []
    nrange = np.arange(int(n_min), int(n_max) + 1)
    nrangex = np.arange(int(n_min), int(n_max) + 2)

    if coef[1] * 2 * x_min + coef[0] < 0: # decreasing
        assert coef[1] * 2 * x_max + coef[0] < 0
        nrange = nrange[::-1]
        nrangex = nrangex[::-1]

    for n in nrange:
        # Solve poly(x) - (2n +1)*pi = 0
        shifted_poly = coef.copy()
        shifted_poly[2] -= (2 * n + 1) * cp.pi
        roots = togpu(np.roots(tocpu(shifted_poly)))
        # Filter real roots within [x_min, x_max]
        real_roots = roots[cp.isreal(roots)].real
        valid_roots = real_roots[(real_roots >= x_min) & (real_roots <= x_max)]
        assert len(valid_roots) == 1
        assert - np.pi <= np.polyval(shifted_poly, valid_roots[0]) <= np.pi
        crossings.extend(valid_roots.tolist())
    # slope: is strictly increasing or decreasing
    for i in range(len(crossings) - 1):
        xv = (crossings[i] + crossings[i + 1]) / 2
        # print( - np.pi + 2 * np.pi * nrangex[i + 1] , np.polyval(coef, xv) , np.pi + 2 * np.pi * nrangex[i + 1])
        assert - np.pi + 2 * np.pi * nrangex[i + 1] <= np.polyval(coef, xv) <= np.pi + 2 * np.pi * nrangex[i + 1]

    assert len(crossings) == len(nrangex) - 1
    return crossings, nrangex


def merge_crossings(crossings_poly1, crossing_n1, crossings_poly2, crossing_n2, x_min, x_max, tol=1e-12):
    # Combine the intersection points and sort them
    merged_crossings = sorted(set(crossings_poly1 + crossings_poly2))

    # Remove crossings that are too close to each other
    filtered_crossings = [merged_crossings[0]]
    for x in merged_crossings[1:]:
        if abs(x - filtered_crossings[-1]) >= tol:
            filtered_crossings.append(x)

    # Initialize the weight list for the filtered sections
    merged_weights = []

    # Iterate through the filtered_crossings to compute the weights for each segment
    sections = []
    for i in range(len(filtered_crossings) + 1):
        # Determine the section bounds
        if i == 0:
            x_left = x_min  # Left of the first intersection
            x_right = filtered_crossings[i]
        elif i == len(filtered_crossings):
            x_left = filtered_crossings[i - 1]
            x_right = x_max  # Right of the last intersection
        else:
            x_left = filtered_crossings[i - 1]
            x_right = filtered_crossings[i]
        sections.append([x_left, x_right])

        # Determine the weights for the current section from crossing_n1 and crossing_n2
        weight1 = next((w for x, w in zip(crossings_poly1, crossing_n1) if x_left < x <= x_right), crossing_n1[-1])
        weight2 = next((w for x, w in zip(crossings_poly2, crossing_n2) if x_left < x <= x_right), crossing_n2[-1])

        merged_weights.append((weight1, weight2))

    assert len(sections) == len(merged_weights)
    return sections, merged_weights



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
    crossings_poly1, crossing_n1 = find_crossings(coefa, x_min, x_max)
    crossings_poly2, crossing_n2 = find_crossings(coefb, x_min, x_max)

    # Combine and sort all crossing points
    sections, all_weights = merge_crossings(crossings_poly1, crossing_n1, crossings_poly2, crossing_n2, x_min, x_max)
    assert len(sections) == len(all_weights)

    for i, (x1, x2) in enumerate(sections):
        xv = (x1 + x2) / 2
        print( - np.pi + 2 * np.pi * all_weights[i + 1][0] , np.polyval(coefa, xv) , np.pi + 2 * np.pi * all_weights[i + 1][0])
        print( - np.pi + 2 * np.pi * all_weights[i + 1][1] , np.polyval(coefb, xv) , np.pi + 2 * np.pi * all_weights[i + 1][1])
        assert - np.pi + 2 * np.pi * all_weights[i + 1][0] <= np.polyval(coefa, xv) <= np.pi + 2 * np.pi * all_weights[i + 1][0]
        assert - np.pi + 2 * np.pi * all_weights[i + 1][1] <= np.polyval(coefb, xv) <= np.pi + 2 * np.pi * all_weights[i + 1][1]

    intersection_points = []
    for idx, ((x_start, x_end), (n1, n2)) in enumerate(zip(sections, all_weights)):
        print(idx, n1, n2)

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
        # Generate x values for plotting

        num_points = 500
        x_vals = np.linspace(x_start, x_end, num_points)
        y1_wrapped = np.polyval(poly1_shifted_coef, x_vals)
        y2_wrapped = np.polyval(poly2_shifted_coef, x_vals)

        # Add shifted polynomials to the subplot
        fig = pltfig(((x_vals, y1_wrapped), (x_vals, y2_wrapped)), addvline=(x_start, x_end), addhline=(-np.pi, np.pi), title=f"{idx=} finditx")

        # Highlight intersection points within this section
        for x_int, y_int in valid_roots:
            fig.add_trace(
                go.Scatter(x=x_int, y=y_int, mode='markers', name='Intersections',
                           marker=dict(color='red', size=8, symbol='circle')),
            )
        # fig.update_yaxes(range=[-np.pi * 5 - 0.1, np.pi * 5 + 0.1])
        fig.show()

    # Print the intersection points
    print("Intersection Points within the specified range:")
    for idx, (x, y) in enumerate(intersection_points, 1):
        print(f"{idx}: x = {x:.12f}, y = {y:.12f}")

    return intersection_points, poly1_shifted_coef, poly2_shifted_coef
