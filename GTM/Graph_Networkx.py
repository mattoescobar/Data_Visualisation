import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy as sp
import scipy.io  as sio
from scipy.sparse import spdiags, coo_matrix
import louvain as lo

## Now the layout function
def forceatlas2_layout(G, iterations=10, linlog=False, pos=None, nohubs=False,
                       kr=0.001, k=None, dim=2):
    """
    Options values are
    g                The graph to layout
    iterations       Number of iterations to do
    linlog           Whether to use linear or log repulsion
    random_init      Start with a random position
                     If false, start with FR
    avoidoverlap     Whether to avoid overlap of points
    degreebased      Degree based repulsion
    """
    # We add attributes to store the current and previous convergence speed
    for n in G:
        G.node[n]['prevcs'] = 0
        G.node[n]['currcs'] = 0
        # To numpy matrix
    # This comes from the spares FR layout in nx
    A = nx.to_scipy_sparse_matrix(G, dtype='f')
    nnodes, _ = A.shape

    try:
        A = A.tolil()
    except Exception as e:
        A = (coo_matrix(A)).tolil()
    if pos is None:
        pos = np.asarray(np.random.random((nnodes, dim)), dtype=A.dtype)
    else:
        pos = pos.astype(A.dtype)
    if k is None:
        k = np.sqrt(1.0 / nnodes)
        # Iterations
    # the initial "temperature" is about .1 of domain area (=1x1)
    # this is the largest step allowed in the dynamics.
    t = 0.1
    # simple cooling scheme.
    # linearly step down by dt on each iteration so last iteration is size dt.
    dt = t / float(iterations + 1)
    displacement = np.zeros((dim, nnodes))
    for iteration in range(iterations):
        displacement *= 0
        # loop over rows
        for i in range(A.shape[0]):
            # difference between this row's node position and all others
            delta = (pos[i] - pos).T
            # distance between points
            distance = np.sqrt((delta ** 2).sum(axis=0))
            # enforce minimum distance of 0.01
            distance = np.where(distance < 0.01, 0.01, distance)
            # the adjacency matrix row
            Ai = np.asarray(A.getrowview(i).toarray())
            # displacement "force"
            Dist = k * k / distance ** 2
            if nohubs:
                Dist = Dist / float(Ai.sum(axis=1) + 1)
            if linlog:
                Dist = np.log(Dist + 1)
            displacement[:, i] += \
                (delta * (Dist - Ai * distance / k)).sum(axis=1)
            # update positions
        length = np.sqrt((displacement ** 2).sum(axis=0))
        length = np.where(length < 0.01, 0.01, length)
        pos += (displacement * t / length).T
        # cool temperature
        t -= dt
        # Return the layout
    return dict(zip(G, pos))


# Loading similarity matrix structure
sim_mat_structure = np.load("similarity_matrix30.npz")  # python
simple_sm = sim_mat_structure["arr_0"]
simple_sm[simple_sm < 0.2] = 0
matlab_sm_structure = sio.loadmat("AM_mm_test_1.mat")
matlab_sm = matlab_sm_structure['adjmm'][0:900, 0:900]
# full_sm = sim_mat_structure["arr_1"]
# full_sm[full_sm < 10 ^ -5] = 0


# # Plotting each adjacency matrix for comparison
# [x, y] = np.meshgrid(np.linspace(1, simple_sm.shape[0], simple_sm.shape[0]),
#                      np.linspace(1, simple_sm.shape[0], simple_sm.shape[0]))
# x = np.ravel(x)
# y = np.ravel(y)
# sim_lat = np.array([x, y])
#
# print "Plotting color mesh image..."
# plt.figure()
# plt.pcolormesh(np.reshape(sim_lat[0, :], (simple_sm.shape[0], simple_sm.shape[0])),
#            np.reshape(sim_lat[1, :], (simple_sm.shape[0], simple_sm.shape[0])), simple_sm,
#            cmap='magma', vmin=0, vmax=1)
# plt.colorbar()
# plt.axis([x.min(), x.max(), y.min(), y.max()])
# plt.gca().invert_yaxis()
# plt.show()
#
# print "Plotting color mesh image..."
# plt.figure()
# plt.pcolormesh(np.reshape(sim_lat[0, :], (simple_sm.shape[0], simple_sm.shape[0])),
#            np.reshape(sim_lat[1, :], (simple_sm.shape[0], simple_sm.shape[0])), matlab_sm,
#            cmap='magma', vmin=0, vmax=1)
# plt.colorbar()
# plt.axis([x.min(), x.max(), y.min(), y.max()])
# plt.gca().invert_yaxis()
#
# Generating graph for full similarity matrix
g = nx.from_numpy_matrix(matlab_sm)
#
# # Analyzing the generated graph
# cc = nx.connected_components(g)
# print nx.degree(g)
#
# Drawing the graph using distinct graph layouts
node_color = []
node_color[0:200] = ['r'] * 200
node_color[200:400] = ['b'] * 200
node_color[400:600] = ['g'] * 200
node_color[600:700] = ['m'] * 100
node_color[700:800] = ['y'] * 100
node_color[800:900] = ['c'] * 100
# print "Random layout"
# plt.figure()
# pos = nx.random_layout(g)
# nx.draw_networkx(g, node_color=node_color, with_labels=False, pos=pos)
# print "Spring layout"
# plt.figure()
# pos = nx.spring_layout(g, scale=10.0, iterations=200)
# nx.draw_networkx(g, node_color=node_color, with_labels=False, pos=pos)
# print "Circular layout"
# plt.figure()
# pos = nx.circular_layout(g)
# nx.draw_networkx(g, node_color=node_color, with_labels=False, pos=pos)
# print "Shell layout"
# plt.figure()
# pos = nx.shell_layout(g)
# nx.draw_networkx(g, node_color=node_color, with_labels=False, pos=pos)
# print "Spectral layout"
# plt.figure()
# pos = nx.spectral_layout(g)
# nx.draw_networkx(g, node_color=node_color, with_labels=False, pos=pos)
# plt.show()

lo.find_partition(g)

print "Creating layout"
pos = forceatlas2_layout(g, iterations=1000)
print "drawing network"
nx.draw_networkx(g, node_color=node_color, with_labels=False, pos=pos)
plt.show()
print "Hello"