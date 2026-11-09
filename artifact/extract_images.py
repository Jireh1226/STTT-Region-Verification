"""
Extract and visualize sample runway images from the VerifyGAN dataset.
Also validates the TinyTaxiNet model by running a forward pass.
"""
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# =============================================================
# 1. Load and inspect the HDF5 dataset
# =============================================================
h5_path = 'VerifyGAN/data/SK_DownsampledGANFocusAreaData.h5'
with h5py.File(h5_path, 'r') as f:
    # NOTE: naming is swapped in the file:
    #   y_train = images (10000, 8, 16)
    #   X_train = labels (10000, 3)
    images = f['y_train'][:]   # shape (10000, 8, 16)
    labels = f['X_train'][:]   # shape (10000, 3)

print(f"Images shape: {images.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Image pixel range: [{images.min():.4f}, {images.max():.4f}]")
print(f"Label columns (first 5 rows):\n{labels[:5]}")
print()

# =============================================================
# 2. Find representative images: centered, left, right
# =============================================================
# Labels column 0 is likely CTE (cross-track error)
cte = labels[:, 0]
print(f"CTE range: [{cte.min():.2f}, {cte.max():.2f}]")
print(f"Label col 1 range: [{labels[:,1].min():.2f}, {labels[:,1].max():.2f}]")
print(f"Label col 2 range: [{labels[:,2].min():.2f}, {labels[:,2].max():.2f}]")

# Find closest to centered (CTE ~ 0)
idx_centered = np.argmin(np.abs(cte))
# Find one shifted left (large positive CTE = aircraft right of centerline)
idx_right_of_center = np.argmax(cte)
# Find one shifted right (large negative CTE = aircraft left of centerline)
idx_left_of_center = np.argmin(cte)

print(f"\nCentered image index: {idx_centered}, CTE = {cte[idx_centered]:.4f}")
print(f"Max CTE (aircraft right) index: {idx_right_of_center}, CTE = {cte[idx_right_of_center]:.4f}")
print(f"Min CTE (aircraft left) index: {idx_left_of_center}, CTE = {cte[idx_left_of_center]:.4f}")

# =============================================================
# 3. Plot the three representative images
# =============================================================
fig, axes = plt.subplots(1, 3, figsize=(12, 3))

samples = [
    (idx_centered, f"Centered\nCTE = {cte[idx_centered]:.2f} m"),
    (idx_right_of_center, f"Aircraft Right of Center\nCTE = {cte[idx_right_of_center]:.2f} m"),
    (idx_left_of_center, f"Aircraft Left of Center\nCTE = {cte[idx_left_of_center]:.2f} m"),
]

for ax, (idx, title) in zip(axes, samples):
    img = images[idx]
    im = ax.imshow(img, cmap='gray', vmin=0, vmax=1, aspect='equal')
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('Column (0-15)')
    ax.set_ylabel('Row (0-7)')

plt.suptitle('TinyTaxiNet 8x16 Runway Images from VerifyGAN Dataset', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('sample_runway_images.png', dpi=200, bbox_inches='tight')
print("\nSaved: sample_runway_images.png")

# =============================================================
# 4. Also save a grid of 9 images spanning the CTE range
# =============================================================
sorted_indices = np.argsort(cte)
# Pick 9 evenly spaced from sorted order
pick_indices = sorted_indices[np.linspace(0, len(sorted_indices)-1, 9, dtype=int)]

fig, axes = plt.subplots(3, 3, figsize=(10, 6))
for i, (ax, idx) in enumerate(zip(axes.flat, pick_indices)):
    ax.imshow(images[idx], cmap='gray', vmin=0, vmax=1, aspect='equal')
    ax.set_title(f"CTE = {cte[idx]:.2f} m", fontsize=9)
    ax.tick_params(labelsize=7)

plt.suptitle('TinyTaxiNet Images Across CTE Range', fontsize=13)
plt.tight_layout()
plt.savefig('runway_images_grid.png', dpi=200, bbox_inches='tight')
print("Saved: runway_images_grid.png")

# =============================================================
# 5. Load and evaluate TinyTaxiNet with Marabou
# =============================================================
print("\n" + "="*60)
print("LOADING TINYTAXINET WITH MARABOU 2.0")
print("="*60)

try:
    from maraboupy import Marabou

    nnet_path = 'VerifyGAN/models/TinyTaxiNet.nnet'
    net = Marabou.read_nnet(nnet_path)

    print(f"Network loaded successfully from: {nnet_path}")
    print(f"Number of input variables: {net.inputVars[0].shape}")
    print(f"Number of output variables: {net.outputVars[0].shape}")

    # Evaluate on the centered image
    centered_img = images[idx_centered].flatten()
    print(f"\nInput (centered image, flattened): shape={centered_img.shape}")
    print(f"  Pixel range: [{centered_img.min():.4f}, {centered_img.max():.4f}]")

    # Forward pass through the network
    output = net.evaluate([centered_img])
    output_vals = np.array(output).flatten()
    print(f"\nNetwork output for centered image:")
    print(f"  y0 (CTE prediction): {float(output_vals[0]):.4f} m")
    print(f"  y1 (HE prediction):  {float(output_vals[1]):.4f} deg")
    print(f"  Ground truth CTE:    {cte[idx_centered]:.4f} m")

    # Evaluate on left and right images too
    for name, idx in [("right-of-center", idx_right_of_center), ("left-of-center", idx_left_of_center)]:
        img_flat = images[idx].flatten()
        out = net.evaluate([img_flat])
        out_vals = np.array(out).flatten()
        print(f"\nNetwork output for {name} image (true CTE={cte[idx]:.2f}):")
        print(f"  y0 (CTE prediction): {float(out_vals[0]):.4f} m")
        print(f"  y1 (HE prediction):  {float(out_vals[1]):.4f} deg")

    print("\n✓ Marabou 2.0 setup validated successfully!")

except ImportError as e:
    print(f"Could not import maraboupy: {e}")
    print("Try: pip3 install maraboupy")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
