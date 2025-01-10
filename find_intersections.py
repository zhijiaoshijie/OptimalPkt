import numpy as np
import plotly.graph_objects as go
from math import ceil, floor

# Define the coefficients for the two polynomials
coeflist = [
    [1.26581062e+08, -6.65625087e+06, 8.42679006e+04],
    [1.26581062e+08, -9.20875231e+06, 1.64245710e+05]
]


# Define the start position and the range around it
start_pos_all_new = 0.030318928843199852
epsilon = 1e-6

# Function to find crossing points where poly(x) = (2n +1)*pi
def find_crossings(coef, x_min, x_max):
    y_min = np.polyval(coef, x_min)
    y_max = np.polyval(coef, x_max)
    y_lower = min(y_min, y_max)
    y_upper = max(y_min, y_max)

    # Compute the range of n values
    n_min = ceil((y_lower - np.pi) / (2 * np.pi))
    n_max = floor((y_upper - np.pi) / (2 * np.pi))

    crossings = []
    for n in range(int(n_min), int(n_max) + 1):
        # Solve poly(x) - (2n +1)*pi = 0
        shifted_poly = (coef[0], coef[1], coef[2] - (2 * n + 1) * np.pi)
        roots = np.roots(shifted_poly)
        # Filter real roots within [x_min, x_max]
        real_roots = roots[np.isreal(roots)].real
        valid_roots = real_roots[(real_roots >= x_min) & (real_roots <= x_max)]
        crossings.extend(valid_roots.tolist())
    return crossings

# Function to determine n for wrapping within a section
def determine_n(coef, x_start, x_end):
    # Choose the midpoint to determine n
    x_mid = (x_start + x_end) / 2
    y_mid = np.polyval(coef, x_mid)
    n = floor((y_mid + np.pi) / (2 * np.pi))
    return n


def find_intersections(coeflist, start_pos_all_new, epsilon):
    x_min = start_pos_all_new - epsilon
    x_max = start_pos_all_new + epsilon

    # Find all crossing points for both polynomials
    crossings_poly1 = find_crossings(coeflist[0], x_min, x_max)
    crossings_poly2 = find_crossings(coeflist[1], x_min, x_max)

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
    from plotly.subplots import make_subplots

    # Determine number of sections to create subplots
    num_sections = len(sections)
    # For better visualization, arrange subplots in rows
    fig = make_subplots(rows=num_sections, cols=1, shared_xaxes=True,
                        subplot_titles=[f"Section {i + 1}: x ∈ [{sec[0]:.8f}, {sec[1]:.8f}]" for i, sec in
                                        enumerate(sections)],
                        vertical_spacing=0.05)

    # Iterate over each section to find intersections and plot
    for idx, (x_start, x_end) in enumerate(sections):
        # Determine n for each polynomial
        n1 = determine_n(coeflist[0], x_start, x_end)
        n2 = determine_n(coeflist[1], x_start, x_end)
        print(idx, n1, n2)

        # Adjust the constant term for wrapping
        # Create shifted polynomials by adding (2n -1)*pi to the constant term
        coef = coeflist[0]
        poly1_shifted_coef = (coef[0], coef[1], coef[2] - (2 * n1) * np.pi)

        coef = coeflist[1]
        poly2_shifted_coef = (coef[0], coef[1], coef[2] - (2 * n2) * np.pi)

        # Compute the difference polynomial
        poly_diff = np.polysub(poly1_shifted_coef, poly2_shifted_coef)

        # Find roots of the difference polynomial
        roots = np.roots(poly_diff)
        # Filter real roots
        real_roots = roots[np.isreal(roots)].real
        # Filter roots within the current section
        valid_roots = real_roots[(real_roots >= x_start) & (real_roots <= x_end)]
        # Convert to float and remove duplicates
        valid_roots = valid_roots.astype(float)

        # Add valid roots to intersection points
        for root in valid_roots:
            y_val = np.polyval(poly1_shifted_coef, root)
            intersection_points.append((root, y_val))

        # Generate x values for plotting
        num_points = 500
        x_vals = np.linspace(x_start, x_end, num_points)
        y1_wrapped = np.polyval(poly1_shifted_coef, x_vals)
        y2_wrapped = np.polyval(poly2_shifted_coef, x_vals)

        # Add shifted polynomials to the subplot
        fig.add_trace(
            go.Scatter(x=x_vals, y=y1_wrapped, mode='lines', name='Poly1 Shifted',
                       line=dict(color='blue')),
            row=idx + 1, col=1
        )
        fig.add_trace(
            go.Scatter(x=x_vals, y=y2_wrapped, mode='lines', name='Poly2 Shifted',
                       line=dict(color='green')),
            row=idx + 1, col=1
        )

        # Add horizontal lines y=-pi and y=pi
        fig.add_trace(
            go.Scatter(x=[x_start, x_end], y=[-np.pi, -np.pi],
                       mode='lines', line=dict(color='gray', dash='dash'), showlegend=False),
            row=idx + 1, col=1
        )
        fig.add_trace(
            go.Scatter(x=[x_start, x_end], y=[np.pi, np.pi],
                       mode='lines', line=dict(color='gray', dash='dash'), showlegend=False),
            row=idx + 1, col=1
        )

        # Add vertical lines at section boundaries
        fig.add_trace(
            go.Scatter(x=[x_start, x_start], y=[-np.pi, np.pi],
                       mode='lines', line=dict(color='black', dash='solid'), showlegend=False),
            row=idx + 1, col=1
        )
        fig.add_trace(
            go.Scatter(x=[x_end, x_end], y=[-np.pi, np.pi],
                       mode='lines', line=dict(color='black', dash='solid'), showlegend=False),
            row=idx + 1, col=1
        )

        # Highlight intersection points within this section
        section_intersections = [(x, y) for (x, y) in intersection_points if x_start <= x <= x_end]
        if section_intersections:
            x_int, y_int = zip(*section_intersections)
            fig.add_trace(
                go.Scatter(x=x_int, y=y_int, mode='markers', name='Intersections',
                           marker=dict(color='red', size=8, symbol='circle')),
                row=idx + 1, col=1
            )

    # Update layout for better visualization
    fig.update_layout(
        height=300 * num_sections,  # Adjust height based on number of sections
        width=800,
        title_text='Intersections of Wrapped Polynomials Near Start Position',
        showlegend=True
    )

    # Adjust y-axis range for all subplots
    for i in range(1, num_sections + 1):
        fig.update_yaxes(range=[-np.pi*5 - 0.1, np.pi*5 + 0.1], row=i, col=1)

    # Show the plot
    fig.show()

    # Print the intersection points
    print("Intersection Points within the specified range:")
    for idx, (x, y) in enumerate(intersection_points, 1):
        print(f"{idx}: x = {x:.12f}, y = {y:.12f}")
