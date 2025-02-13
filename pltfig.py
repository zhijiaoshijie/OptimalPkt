import plotly.express as px
import plotly.graph_objects as go
from utils import tocpu, togpu, sqlist
import numpy as np
def pltfig(datas, title = None, yaxisrange = None, modes = None, marker = None, addvline = None, addhline = None, line_dash = None, fig = None, line=None):
    """
    Plot a figure with the given data and parameters.

    Parameters:
    datas : list of tuples
        Each tuple contains two lists or array-like elements, the data for the x and y axes.
        If only y data is provided, x data will be generated using np.arange.
    title : str, optional
        The title of the plot (default is None).
    yaxisrange : tuple, optional
        The range for the y-axis as a tuple (min, max) (default is None).
    mode : str, optional
        The mode of the plot (e.g., 'line', 'scatter') (default is None).
    marker : str, optional
        The marker style for the plot (default is None).
    addvline : float, optional
        The x-coordinate for a vertical line (default is None).
    addhline : float, optional
        The y-coordinate for a horizontal line (default is None).
    line_dash : str, optional
        The dash style for the line (default is None).
    fig : matplotlib.figure.Figure, optional
        The figure object to plot on (default is None).
    line : matplotlib.lines.Line2D, optional
        The line object for the plot (default is None).

    Returns:
    None
    """
    if fig is None: fig = go.Figure(layout_title_text=title)
    elif title is not None: fig.update_layout(title_text=title)
    if not all(len(data) == 2 for data in datas): datas = [(np.arange(len(data)), data) for data in datas]
    if modes is None:
        modes = ['lines' for _ in datas]
    elif isinstance(modes, str):
        modes = [modes for _ in datas]
    for idx, ((xdata, ydata), mode) in enumerate(zip(datas, modes)):
        if line == None and idx == 1: line = dict(dash='dash')
        fig.add_trace(go.Scatter(x=tocpu(sqlist(xdata)), y=tocpu(sqlist(ydata)), mode=mode, marker=marker, line=line))
        assert len(tocpu(sqlist(xdata))) == len(tocpu(sqlist(ydata)))
    pltfig_hind(addhline, addvline, line_dash, fig, yaxisrange)
    return fig



def pltfig1(xdata, ydata, title = None, yaxisrange = None, mode = None, marker = None, addvline = None, addhline = None, line_dash = None, fig = None, line=None):
    """
    Plot a figure with the given data and parameters.

    Parameters:
    xdata : list or array-like or None
        The data for the x-axis.
        If is None, and only y data is provided, x data will be generated using np.arange.
    ydata : list or array-like
        The data for the y-axis.
    title : str, optional
        The title of the plot (default is None).
    yaxisrange : tuple, optional
        The range for the y-axis as a tuple (min, max) (default is None).
    mode : str, optional
        The mode of the plot (e.g., 'line', 'scatter') (default is None).
    marker : str, optional
        The marker style for the plot (default is None).
    addvline : float, optional
        The x-coordinate for a vertical line (default is None).
    addhline : float, optional
        The y-coordinate for a horizontal line (default is None).
    line_dash : str, optional
        The dash style for the line (default is None).
    fig : matplotlib.figure.Figure, optional
        The figure object to plot on (default is None).
    line : matplotlib.lines.Line2D, optional
        The line object for the plot (default is None).

    Returns:
    None
    """
    if xdata is None: xdata = np.arange(len(ydata))
    if fig is None: fig = go.Figure(layout_title_text=title)
    elif title is not None: fig.update_layout(title_text=title)
    if mode is None: mode = 'lines'
    fig.add_trace(go.Scatter(x=tocpu(sqlist(xdata)), y=tocpu(sqlist(ydata)), mode=mode, marker=marker, line=line))
    assert len(tocpu(sqlist(xdata))) == len(tocpu(sqlist(ydata)))
    pltfig_hind(addhline, addvline, line_dash, fig, yaxisrange)
    return fig

def pltfig_hind(addhline, addvline, line_dash_in, fig, yaxisrange):
    if yaxisrange: fig.update_layout(yaxis=dict(range=yaxisrange), )
    if addvline is not None:
        if line_dash_in is None:
            line_dash = ['dash' for _ in range(len(addvline))]
            line_dash[0] = 'dot'
        elif isinstance(line_dash_in, str):
            line_dash = [line_dash_in for _ in range(len(addvline))]
        else: line_dash = line_dash_in
        for x, ldash in zip(addvline, line_dash): fig.add_vline(x=x, line_dash=ldash)
    if addhline is not None:
        if line_dash_in is None:
            line_dash = ['dash' for _ in range(len(addhline))]
            line_dash[0] = 'dot'
        elif isinstance(line_dash_in, str):
            line_dash = [line_dash_in for _ in range(len(addhline))]
        else: line_dash = line_dash_in
        for y, ldash in zip(addhline, line_dash): fig.add_hline(y=y, line_dash=ldash)
