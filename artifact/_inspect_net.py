import os, sys
from pathlib import Path
os.chdir(Path(__file__).resolve().parent)
from maraboupy import Marabou
from maraboupy import MarabouNetworkONNX
import numpy as np

net = Marabou.read_onnx('composed_centered_with_sum.onnx')
print('Type:', type(net).__name__)
print('inputVars shape:', np.array(net.inputVars[0]).shape)
print('outputVars:', np.array(net.outputVars[0]).flatten())
print('numVars:', net.numVars)
print()

# Maraboupy's ONNX backend stores variable->tensor mapping. Let's look for it.
print('Has varMap?', hasattr(net, 'varMap'))
print('Has shapeMap?', hasattr(net, 'shapeMap'))
print('Has madeGraphEquations?', hasattr(net, 'madeGraphEquations'))
print('Has constantMap?', hasattr(net, 'constantMap'))

# Try internal _names
internal_attrs = [a for a in dir(net) if 'map' in a.lower() or 'name' in a.lower() or 'tensor' in a.lower()]
print('Internal attrs:')
for a in internal_attrs:
    print(' ', a)

# Probably node-based interface
print()
print('getNode signature?')
import inspect
try:
    sig = inspect.signature(net.getNode)
    print(f'  getNode{sig}')
except Exception as e:
    print(f'  err: {e}')
try:
    sig = inspect.signature(net.splitNetworkAtNode)
    print(f'  splitNetworkAtNode{sig}')
except Exception as e:
    print(f'  err: {e}')

# Check if read_onnx has an 'output' parameter to expose intermediate variables
import maraboupy.Marabou as M
print()
print('read_onnx signature:')
sig = inspect.signature(M.read_onnx)
print(f'  read_onnx{sig}')

# Inspect MarabouNetworkONNX
print()
print('MarabouNetworkONNX module attrs:')
for a in dir(MarabouNetworkONNX):
    if not a.startswith('_'):
        print(' ', a)

# Try reading the network with the decoded pixels exposed as an extra output
print()
print('=== Re-reading networks with extra outputs ===')
pixels_tensor = 'StatefulPartitionedCall/composed_with_sum/pixels/BiasAdd:0'
combined_tensor = 'combined_out'
for subset in ['centered', 'left', 'right', 'off_center']:
    print(f'subset={subset}:')
    try:
        net2 = Marabou.read_onnx(f'composed_{subset}_with_sum.onnx', outputNames=[combined_tensor, pixels_tensor])
        print(f'  numVars: {net2.numVars}, output count: {len(net2.outputVars)}')
        for i, ov in enumerate(net2.outputVars):
            arr = np.array(ov)
            print(f'    [{i}] shape={arr.shape}, first/last={arr.flatten()[0]}/{arr.flatten()[-1]}')
    except Exception as e:
        print(f'  Error: {type(e).__name__}: {e}')

# Sanity solve: tiny query just to confirm decoded pixel variables behave properly
print()
print('=== Sanity solve at a tight latent box, no math constraint ===')
import time
net3 = Marabou.read_onnx('composed_centered_with_sum.onnx', outputNames=[combined_tensor, pixels_tensor])
inp = np.array(net3.inputVars[0]).flatten()
ttn_out = np.array(net3.outputVars[0]).flatten()
pix_out = np.array(net3.outputVars[1]).flatten()
print(f'input vars: {len(inp)}, ttn vars: {ttn_out}, pixel vars: {pix_out[0]}..{pix_out[-1]} (n={len(pix_out)})')

# Tight latent box (around origin)
for i in inp:
    net3.setLowerBound(int(i), -0.5)
    net3.setUpperBound(int(i), 0.5)

# Force CTE >= 10 (likely UNSAT)
net3.setLowerBound(int(ttn_out[0]), 10.0)
opts = Marabou.createOptions(verbosity=0, timeoutInSeconds=30)
t0 = time.time()
status, vals, _ = net3.solve(verbose=False, options=opts)
print(f'  status: {status!r}, t={time.time()-t0:.2f}s')

# Now try with CTE >= 0 (probably SAT, we can check the pixels variable values)
net4 = Marabou.read_onnx('composed_centered_with_sum.onnx', outputNames=[combined_tensor, pixels_tensor])
inp = np.array(net4.inputVars[0]).flatten()
ttn_out = np.array(net4.outputVars[0]).flatten()
pix_out = np.array(net4.outputVars[1]).flatten()
for i in inp:
    net4.setLowerBound(int(i), -0.5); net4.setUpperBound(int(i), 0.5)
net4.setLowerBound(int(ttn_out[0]), 0.0)
opts = Marabou.createOptions(verbosity=0, timeoutInSeconds=30)
t0 = time.time()
status, vals, _ = net4.solve(verbose=False, options=opts)
print(f'  small SAT? status={status!r}, t={time.time()-t0:.2f}s')
if status.strip().lower() == 'sat':
    pixels_vals = np.array([vals[int(v)] for v in pix_out])
    print(f'  decoded pixels: shape={pixels_vals.shape}, sum={pixels_vals.sum():.4f}')
    print(f'  pixel_sum output: {vals[int(ttn_out[2])]:.4f}')
    print(f'  CTE: {vals[int(ttn_out[0])]:.4f}, HE: {vals[int(ttn_out[1])]:.4f}')

import onnx
# Inspect ONNX structure to find decoded pixel layer node
model = onnx.load('composed_centered_with_sum.onnx')
print('Inputs:')
for inp in model.graph.input:
    print(f'  {inp.name} shape={[d.dim_value for d in inp.type.tensor_type.shape.dim]}')
print('Outputs:')
for out in model.graph.output:
    print(f'  {out.name} shape={[d.dim_value for d in out.type.tensor_type.shape.dim]}')
print()
print('Nodes (op_type, name, inputs->outputs):')
for i, node in enumerate(model.graph.node):
    print(f'  {i:3d}  {node.op_type:15s}  name={node.name}')
    print(f'         inputs:  {list(node.input)}')
    print(f'         outputs: {list(node.output)}')

# Look for "pixels" tensor as an output of some node, then we want to expose it.
print()
print('Looking for "pixels"-related tensors:')
for node in model.graph.node:
    for o in node.output:
        if 'pixel' in o.lower() or 'pixels' in o.lower() or 'dec' in o.lower():
            print(f'  output tensor: {o}  (from {node.op_type} {node.name})')
