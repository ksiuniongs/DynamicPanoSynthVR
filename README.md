# PanoSynthVR
PanoSynthVR: View Synthesis From A Single Input Panorama with Multi-Cylinder Images

Code from our papers:

John Waidhofer, Richa Gadgil, Anthony Dickson, Stefanie Zollmann, and Jonathan Ventura.  PanoSynthVR: Toward Light-weight 360-Degree View Synthesis from a Single Panoramic Input.  IEEE International Symposium on Mixed and Augmented Reality. 2022.

Richa Gadgil, Reesa John, Stefanie Zollmann, and Jonathan Ventura.  PanoSynthVR: View Synthesis From A Single Input Panorama with Multi-Cylinder Images.  ACM SIGGRAPH 2021 Posters.

# Conda environment

To create and activate the conda environment:

    conda env create
    conda activate panosynthvr

Then run download_mpi.sh to get the MPI code and pre-trained weights.


Run MCI generation only
```
  python generate_mci.py --input docs/assets/matterport2k/0/input.png --width 2048 --height 1024 --o outputfolder
```
Run MCI generation only and display output in website using WebXR
```
  python generate_mci.py --input example.jpg --width 2048 --height 1024 --o outputfolder --s 1
```
docs/assets/matterport2k/0/input.png

http://127.0.0.1:8000/docs/atlas_sequence_viewer.html?scene=campus_run_1024

http://127.0.0.1:8000/docs/atlas_sequence_viewer.html?scene=campus_run_1024_dy&foreground=foreground_rgba&foreground_depth=1.02&bg_static=true


http://127.0.0.1:8000/docs/atlas_sequence_viewer.html?scene=campus_run_1024_dy_1_interval_5&foreground=foreground_rgba&foreground_depth=1.0&bg_static=true

http://127.0.0.1:8000/docs/atlas_sequence_viewer.html?scene=campus_run_2048_5&manifest=frames_manifest.json&foreground_manifest=foreground_manifest.json

```
http://127.0.0.1:3600/docs/renderer.html?mode=video_atlas&scene=-1&name=<scene_name>
```

http://localhost:8000/docs/atlas_sequence_viewer.html?scene=my_atlases&manifest=frames_manifest.json&fps=24&loop=true
