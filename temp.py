import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Sample data
xval = list(range(10))
yval1= [1, 3, 2, 5, 7, 8, 6, 9, 11, 12]
yval2= [2, 4, 1, 6, 8, 7, 5, 10, 12, 13]

fig = px.line(y=[yval1, yval2],color_discrete_sequence=['blue', 'red'], title="input data 15 symbol")
fig.data[0].name = 'Line 1'
fig.data[1].name = 'Line 2'
fig.show()
fig.write_html("input_data.html")
fig.write_html(os.path.join(Config.figpath, f"resarray {tid=} {est_code=}.html"))
fig.add_vline(x=est_code, line=dict(color='black', width=2, dash='dash'), annotation_text='est_code',
              annotation_position='top')
