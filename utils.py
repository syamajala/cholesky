import math
import plotly.graph_objs as go
from plotly.offline import plot


def depth(dim, size=64):
    return math.ceil(math.log(dim/size)/math.log(2))+1


def leaf_size(dim, depth):
    return dim/2**(depth-1)


def subregions(depth):
    s = 0
    for i in reversed(range(0, depth)):
        s += (i+1)*2**i
    return s


def tree(levels):
    t = []

    end = None
    start = 1
    for level in range(levels, -1, -1):
        if end is None:
            end = 2**level+1
        else:
            end = start + 2**level
            seps = list(range(start, end))
            t.append(seps)
            start = end

    return dict(enumerate(t))


def print_tree_size(t):
    for lvl, nodes in t.items():
        print("Level:", lvl, "Nodes:", len(nodes))


dim = 125000
depths = list(range(1, depth(dim)+1))
leaf_sizes = list(map(lambda x: leaf_size(dim, x), depths))
num_subregions = list(map(lambda x: subregions(x), depths))
num_leaves = list(map(lambda x: 2**(x-1), depths))

print("Depths:", depths)
print("Leaf sizes:", leaf_sizes)
print("Subregions:", num_subregions)

traces = [go.Scatter(x=depths, y=leaf_sizes, mode='lines+markers', name='Depth vs Block Size'),
          go.Scatter(x=depths, y=num_subregions, mode='lines+markers', name="Depth vs Num Subregions", yaxis='y2'),
          go.Scatter(x=depths, y=num_leaves, mode='lines+markers', name="Depth vs Num Leaf Subregions", yaxis='y2')]
layout = go.Layout(title='50^3 Laplacian Depth vs Block Size',
                   xaxis={'title': "Depth"},
                   yaxis={'title': "Block Size"},
                   yaxis2={'title': "Num Subregions",
                           'overlaying': 'y',
                           'side': 'right'})
fig = {'data': traces, 'layout': layout}
plot(fig)
