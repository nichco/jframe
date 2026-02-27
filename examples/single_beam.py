import numpy as np
import jframe as jf


# create a 1D beam 1 mesh
num_nodes_1 = 21
beam_1_mesh = np.zeros((num_nodes_1, 3))
beam_1_mesh[:, 1] = np.linspace(0, 10, num_nodes_1)

# create beam 1 loads
beam_1_loads = np.zeros((num_nodes_1, 6))
beam_1_loads[:, 2] = 20000

# create cs properties for beam 1
beam_1_cs = jf.CSTube()

# create beam 1 with boundary conditions and loads
beam_1 = jf.Beam(name='beam_1', mesh=beam_1_mesh, cs=beam_1_cs, E=E, G=G, rho=rho)
beam_1.fix(node=0)
beam_1.add_load(beam_1_loads)


frame = jf.Frame(beams=[beam_1])
frame.solve()