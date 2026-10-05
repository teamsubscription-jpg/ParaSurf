# **Docker Installation 🐳**

The image is built from this repository and already contains the model weights
(`weights/Paragraph_expanded_best.pth`, `Pecan_best.pth`, `Paragraph_expanded_heavy_best.pth`, `Paragraph_expanded_light_best.pth`).
By default it starts the RunPod serverless worker (`handler.py`).

## 🚀 Build the Docker Environment
```bash
sudo docker build -t parasurf:latest .
```
## 🏃‍♂️ Run the Docker Container interactively
```bash
sudo docker run --gpus all -it --rm \
    --name parasurf_container \
    parasurf:latest bash
```

## Run a prediction 🔍
```bash
python blind_predict.py --receptor test_blind_prediction/4N0Y_receptor_1.pdb --model_weights weights/Paragraph_expanded_best.pth
```

## ☁️ RunPod serverless
Deploy the repository as a serverless endpoint; the worker accepts jobs like:
```json
{"input": {"pdb": "<contents of an antibody .pdb file>", "model": "paragraph_expanded", "mesh_dense": 0.3}}
```
`model` is one of `paragraph_expanded` (default), `pecan`, `heavy`, `light`. Set `"return_pdb": false` to get only the per-residue scores.
