
import sys
import os
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from cv2 import imwrite

# Add project root to path for single_view_mpi
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from single_view_mpi.libs import mpi
from single_view_mpi.libs import nets

def load_model(weights_path=None):
    """Build MPI model and load pre-trained weights."""
    if weights_path is None:
        weights_path = PROJECT_ROOT / 'single_view_mpi_full_keras' / 'single_view_mpi_keras_weights'
    
    inputs = tf.keras.Input(shape=(None, None, 3))
    output = nets.mpi_from_image(inputs)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.load_weights(str(weights_path))

    return model

def get_layer(layers, n):
    """Convert a single MPI layer to BGRA image."""
    layer = layers[n].numpy()
    layer[:, :, :3] *= layer[:, :, 3:]  # pre-multiply alpha
    layer = (layer * 255).astype('uint8')
    layer = cv2.cvtColor(layer, cv2.COLOR_RGBA2BGRA)
    return layer

def build_atlas(layers, output_width, output_height):
    """Assemble 32 layers into an 8x4 atlas texture."""
    H, W = output_height, output_width
    rows = 8
    cols = 4
    atlas = np.zeros((H * rows, W * cols, 4), dtype='uint8')
    n = 0
    for r in range(rows):
        myr = (rows - 1) - r
        for c in range(cols):
            layer = get_layer(layers, n)
            atlas[H * myr:H * (myr + 1), W * c:W * (c + 1)] = layer[:, ::-1]
            n += 1
    return atlas

def process_frame(model, frame_rgb, depths, output_dir, frame_index,
                  output_width, output_height, build_atlas_output=True):
    """Run MPI inference on a single RGB frame."""
    # Resize to requested dimensions
    frame_rgb = tf.convert_to_tensor(frame_rgb, dtype=tf.float32)
    frame_rgb = tf.image.resize(frame_rgb, (output_height, output_width), method='area')
    input_rgb = frame_rgb.numpy()

    # Cylindrical wrap padding
    height, width = input_rgb.shape[:2]
    padding = width // 4
    left = input_rgb[:, 0:padding]
    right = input_rgb[:, width - padding:width]
    input_rgb_padded = np.concatenate((right, input_rgb, left), axis=1)

    # Generate MPI layers
    layers_padded = model(input_rgb_padded[tf.newaxis])[0]

    # Remove padding
    layers = layers_padded[:, :, padding:-padding, :]

    # Disparity map for debugging / inspection
    disparity = mpi.disparity_from_layers(layers, depths)
    disparity = tf.squeeze(disparity)

    # Prepare output folder for the frame
    frame_folder = os.path.join(output_dir, f'frame_{frame_index:06d}')
    os.makedirs(frame_folder, exist_ok=True)

    # Save resized input for reference
    input_bgr = cv2.cvtColor((input_rgb * 255).astype('uint8'), cv2.COLOR_RGB2BGR)
    imwrite(f'{frame_folder}/input.png', input_bgr)

    # Save disparity map
    imwrite(f'{frame_folder}/disparity_map.png',
            (disparity * 255).numpy().astype('uint8'))

    if build_atlas_output:
        atlas = build_atlas(layers, output_width, output_height)
        imwrite(os.path.join(frame_folder, 'atlas.png'), atlas)
    else:
        os.makedirs(f'{frame_folder}/layers', exist_ok=True)
        for n in range(32):
            layer = get_layer(layers, n)
            imwrite(f'{frame_folder}/layers/layer_{n}.png', layer)

def build_atlas_optimized(layers_tensor, mask, output_width, output_height):
    rows, cols = 8, 4
    H, W = output_height, output_width
    atlas = np.zeros((H * rows, W * cols, 4), dtype=np.uint8)
    n = 0
    
    # mask is (H, W, 1) float32
    
    for r in range(rows):
        myr = (rows - 1) - r
        for c in range(cols):
            # Extract single layer from tensor: shape (H, W, 4)
            # Doing .numpy() here only brings 1 layer into CPU memory (approx 32MB)
            layer = layers_tensor[n].numpy()
            
            # Apply mask to this layer
            layer *= mask
            
            # Pre-multiply alpha
            layer[:, :, :3] *= layer[:, :, 3:]
            
            # Convert to uint8
            layer = (layer * 255).astype(np.uint8)
            
            # RGBA to BGRA
            layer = cv2.cvtColor(layer, cv2.COLOR_RGBA2BGRA)
            
            # Place in atlas (flip horizontally as per original logic)
            atlas[H * myr:H * (myr + 1), W * c:W * (c + 1)] = layer[:, ::-1]
            
            n += 1
            
    return atlas
