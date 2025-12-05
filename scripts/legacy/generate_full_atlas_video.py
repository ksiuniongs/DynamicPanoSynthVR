import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import os
import cv2
import numpy as np
import tensorflow as tf
from argparse import ArgumentParser
from cv2 import imwrite

from single_view_mpi.libs import mpi
from single_view_mpi.libs import nets


def load_model():
    """Build MPI model and load pre-trained weights."""
    inputs = tf.keras.Input(shape=(None, None, 3))
    output = nets.mpi_from_image(inputs)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.load_weights('single_view_mpi_full_keras/single_view_mpi_keras_weights')

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


def pad_to_multiple(image, multiple=128):
    """Pad H/W up to the next multiple of `multiple` via reflection."""
    height, width = image.shape[:2]
    target_height = int(np.ceil(height / multiple)) * multiple
    target_width = int(np.ceil(width / multiple)) * multiple

    pad_bottom = target_height - height
    pad_right = target_width - width

    if pad_bottom or pad_right:
        image = np.pad(
            image,
            ((0, pad_bottom), (0, pad_right), (0, 0)),
            mode='reflect'
        )

    return image, pad_bottom, pad_right


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
    input_rgb_wrapped = np.concatenate((right, input_rgb, left), axis=1)

    # Ensure tensor spatial dims are multiples of 128 for the UNet
    input_rgb_ready, pad_bottom, pad_right = pad_to_multiple(input_rgb_wrapped, 128)

    # Generate MPI layers
    layers_padded = model(input_rgb_ready[tf.newaxis])[0]

    # Remove the helper padding and cylindrical wrap
    if pad_bottom or pad_right:
        layers_padded = layers_padded[:, :input_rgb_wrapped.shape[0], :input_rgb_wrapped.shape[1], :]
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


def generate_from_video(model, video_path, output_path, output_width,
                        output_height, build_atlas_output=True, max_frames=-1):
    """Generate multi-cylinder atlases for each frame in a video."""
    depths = mpi.make_depths(1.0, 100.0, 32).numpy()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f'Unable to open video: {video_path}')

    frame_idx = 0
    while True:
        if max_frames > -1 and frame_idx >= max_frames:
            break

        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb = frame_rgb.astype(np.float32) / 255.0

        process_frame(model=model,
                      frame_rgb=frame_rgb,
                      depths=depths,
                      output_dir=output_path,
                      frame_index=frame_idx,
                      output_width=output_width,
                      output_height=output_height,
                      build_atlas_output=build_atlas_output)

        if (frame_idx + 1) % 10 == 0:
            print(f'Processed {frame_idx + 1} frames')

        frame_idx += 1

    cap.release()
    print(f'Finished processing {frame_idx} frames.')


if __name__ == '__main__':
    parser = ArgumentParser(description='Generate multi-cylinder frames from input panorama video')

    parser.add_argument('--input',
                        required=True,
                        help='input video file')
    parser.add_argument('--width',
                        required=True,
                        type=int,
                        help='output frame width (will resize)')
    parser.add_argument('--height',
                        required=True,
                        type=int,
                        help='output frame height (will resize)')
    parser.add_argument('--output', '-o',
                        required=True,
                        help='output directory to store per-frame folders')
    parser.add_argument('--max_frames',
                        type=int,
                        default=-1,
                        help='optional limit on number of frames to process')
    parser.add_argument('--no_atlas',
                        action='store_true',
                        help='save individual MPI layers instead of atlas')

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    model = load_model()

    generate_from_video(model=model,
                        video_path=args.input,
                        output_path=args.output,
                        output_width=args.width,
                        output_height=args.height,
                        build_atlas_output=not args.no_atlas,
                        max_frames=args.max_frames)
