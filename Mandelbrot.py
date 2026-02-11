import numpy as np
import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
from numba import jit

# JIT-compiled Mandelbrot calculation with parallel processing for massive speedup
@jit(nopython=True, parallel=True, fastmath=True)
def mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    
    # Smooth iteration count for beautiful gradients
    result = np.zeros((height, width))
    
    for i in range(height):
        for j in range(width):
            c = complex(x[j], y[i])
            z = 0.0j
            for n in range(max_iter):
                if abs(z) > 2.0:
                    # Smooth coloring algorithm
                    result[i, j] = n + 1 - np.log(np.log(abs(z))) / np.log(2)
                    break
                z = z * z + c
            else:
                result[i, j] = max_iter
    
    return result

# Vibrant rainbow colormap matching the reference image
colors = ['#2a2a2a', '#1a0a2a', '#2a1a4a', '#4a3a8a', '#6a5aaa',
          '#3a7acc', '#5abaff', '#7ae5ff', '#ffaa55', '#ff7a33',
          '#ff5a88', '#cc3aaa', '#8a5acc', '#5a8aff', '#aae5ff', '#d4b896']
n_bins = 256
cmap = LinearSegmentedColormap.from_list('fractal_rainbow', colors, N=n_bins)

# Figure setup with dark background
fig = plt.figure(figsize=(8, 8), facecolor='black')
ax = fig.add_subplot(111)
ax.set_facecolor('black')
ax.axis("off")

# Balanced resolution for detail + performance
width, height = 500, 500
max_iter = 200  # Good balance of detail and speed

# Zoom parameters - zooming into a beautiful location
zoom_center = (-0.743643887037151, 0.131825904205330)
zoom_speed = 0.95  # Faster zoom for deeper exploration

# Pre-create the image object for faster updates
im = ax.imshow(np.zeros((height, width)), cmap=cmap, 
               extent=[-2, 2, -2, 2], interpolation='bilinear')

# Add title
title = ax.text(0.5, 0.98, 'Mandelbrot Set - Deep Zoom', 
                transform=ax.transAxes, color='white', 
                fontsize=14, ha='center', va='top',
                fontweight='bold', family='monospace')

def update(frame):
    scale = zoom_speed ** frame
    xmin = zoom_center[0] - 1.5 * scale
    xmax = zoom_center[0] + 1.5 * scale
    ymin = zoom_center[1] - 1.5 * scale
    ymax = zoom_center[1] + 1.5 * scale

    # Calculate fractal
    fractal = mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter)
    
    # Update image data (much faster than clearing and redrawing)
    im.set_data(fractal)
    im.set_extent([xmin, xmax, ymin, ymax])
    im.set_clim(vmin=0, vmax=max_iter)
    
    # Update zoom level display
    zoom_level = 1.0 / scale
    title.set_text(f'Mandelbrot Set - Zoom: {zoom_level:.1e}x')
    
    return im, title

# Smooth 30fps animation with 250 frames for deep zoom
ani = FuncAnimation(fig, update, frames=250, interval=33, blit=True)
plt.tight_layout()
print("Animation ready! Opening window...")
plt.show()
print("Animation window closed.")
