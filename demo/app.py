import hydra
from omegaconf import OmegaConf
from flask import Flask, jsonify, request
from flask_cors import CORS
from utils import load_ply
import numpy as np
import torch
import os
import numpy as np
import argparse
from pc_sam.model.pc_sam import PointCloudSAM
from pc_sam.utils.torch_utils import replace_with_fused_layernorm
from pc_sam.model.loss import compute_iou
from safetensors.torch import load_model

parser = argparse.ArgumentParser()
parser.add_argument("--host", type=str, default="localhost")
parser.add_argument("--port", type=int, default=5000)
parser.add_argument("--checkpoint", type=str, default="pretrained/model.safetensors")
parser.add_argument("--pointcloud", type=str, default="temp.ply")
parser.add_argument("--config", type=str, default="large", help="path to config file")
parser.add_argument("--config_dir", type=str, default="../configs")
parser.add_argument("--ckpt_path", type=str, default="./pretrained/model.safetensors")
parser.add_argument(
    "--max_points",
    type=int,
    default=50000,
    help="Numero massimo di punti consentiti per la nuvola (sfoltimento se superato; <= 0 per disabilitare, default: 50000)",
)
parser.add_argument(
    "--voxel_size",
    type=float,
    default=None,
    help="Dimensione del voxel per eventuale voxel downsampling preliminare (opzionale)",
)
args = parser.parse_args()

# PCSAM variables
pc_xyz, pc_rgb = None, None
prompts, labels = [], []
prompt_mask = None
obj_path = None
output_dir = "results"
segment_mask = None
masks = []

# Flask Backend
app = Flask(__name__, static_folder="static")
CORS(
    app, origins=f"{args.host}:{args.port}", allow_headers="Access-Control-Allow-Origin"
)

# change "./pretrained/model.safetensors" to the path of the checkpoint

# ---------------------------------------------------------------------------- #
# Load configuration
# ---------------------------------------------------------------------------- #
with hydra.initialize(args.config_dir, version_base=None):
    cfg = hydra.compose(config_name=args.config)
    OmegaConf.resolve(cfg)
    # print(OmegaConf.to_yaml(cfg))


# ---------------------------------------------------------------------------- #
# Setup model
# ---------------------------------------------------------------------------- #
model = hydra.utils.instantiate(cfg.model)
model.apply(replace_with_fused_layernorm)

# ---------------------------------------------------------------------------- #
# Load pre-trained model
# ---------------------------------------------------------------------------- #
load_model(model, args.ckpt_path)
model = model.cuda()  # SPINGI IL MODELLO SULLA GPU
sam = model
sam.eval()


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/static/<path:path>")
def static_server(path):
    return app.send_static_file(path)


@app.route("/mesh/<path:path>")
def mesh_server(path):
    # path = f"/home/yuchen/workspace/annotator_3d/src/static/models/{path}"
    # print(path)
    # path = "/home/yuchen/workspace/annotator_3d/src/static/models/Rhino/White Rhino.obj"
    path = f"models/{path}"
    print(path)
    return app.send_static_file(path)


@app.route("/sampled_pointcloud", methods=["POST"])
def sampled_pc():
    request_data = request.get_json()
    points = request_data["points"].values()
    points = np.array(list(points)).reshape(-1, 3)
    colors = request_data["colors"].values()
    colors = np.array(list(colors)).reshape(-1, 3)

    global pc_xyz, pc_rgb
    pc_xyz, pc_rgb = (
        torch.from_numpy(points).cuda().float(),
        torch.from_numpy(colors).cuda().float(),
    )
    pc_xyz, pc_rgb = pc_xyz.unsqueeze(0), pc_rgb.unsqueeze(0)
    sam.set_point_cloud(pc_xyz, pc_rgb)

    response = "success"
    return jsonify({"response": response})


@app.route("/pointcloud/<path:path>")
def pointcloud_server(path):
    path = args.pointcloud
    global obj_path
    obj_path = path
    max_pts = args.max_points if args.max_points and args.max_points > 0 else None
    points = load_ply(
        f"./demo/static/models/{path}",
        max_points=max_pts,
        voxel_size=args.voxel_size,
    )
    xyz = points[:, :3]
    rgb = points[:, 3:6] / 255

    # normalize
    shift = xyz.mean(0)
    scale = np.linalg.norm(xyz - shift, axis=-1).max()
    xyz = (xyz - shift) / scale

    # set pcsam variables
    global pc_xyz, pc_rgb
    pc_xyz, pc_rgb = (
        torch.from_numpy(xyz).cuda().float(),
        torch.from_numpy(rgb).cuda().float(),
    )
    pc_xyz, pc_rgb = pc_xyz.unsqueeze(0), pc_rgb.unsqueeze(0)

    # Pre-calcola le feature dell'encoder 3D in VRAM
    print(f"[INFO Caching] Pre-calcolo dell'Encoder 3D in corso per {xyz.shape[0]} punti...")
    sam.set_point_cloud(pc_xyz, pc_rgb)
    print("[INFO Caching] Feature 3D pronte in VRAM! I prompt saranno elaborati all'istante.")

    # flatten
    xyz = xyz.flatten()
    rgb = rgb.flatten()

    return jsonify({"xyz": xyz.tolist(), "rgb": rgb.tolist()})


@app.route("/clear", methods=["POST"])
def clear():
    global prompts, labels, prompt_mask, segment_mask
    prompts, labels = [], []
    prompt_mask = None
    segment_mask = None
    return jsonify({"status": "cleared"})


@app.route("/next", methods=["POST"])
def next():
    global prompts, labels, segment_mask, masks, prompt_mask
    masks.append(segment_mask.cpu().numpy())
    prompts, labels = [], []
    prompt_mask = None
    return jsonify({"status": "cleared"})


@app.route("/save", methods=["POST"])
def save():
    os.makedirs(output_dir, exist_ok=True)
    global pc_xyz, pc_rgb, segment_mask, obj_path, masks
    xyz = pc_xyz[0].cpu().numpy()
    rgb = pc_rgb[0].cpu().numpy()
    masks = np.stack(masks)
    obj_path = obj_path.split(".")[0]
    np.save(f"{output_dir}/{obj_path}.npy", {"xyz": xyz, "rgb": rgb, "mask": masks})
    global prompts, labels, prompt_mask
    prompts, labels = [], []
    prompt_mask = None
    segment_mask = None
    return jsonify({"status": "saved"})


@app.route("/segment", methods=["POST"])
def segment():
    request_data = request.get_json()
    prompt_point = request_data["prompt_point"]
    prompt_label = request_data["prompt_label"]

    # append prompt
    global prompts, labels, prompt_mask
    prompts.append(prompt_point)
    labels.append(prompt_label)

    prompt_points = torch.from_numpy(np.array(prompts)).cuda().float()[None, ...]
    prompt_labels = torch.from_numpy(np.array(labels)).cuda()[None, ...]

    data = {
        "points": pc_xyz,
        "rgb": pc_rgb,
        "prompt_points": prompt_points,
        "prompt_labels": prompt_labels,
        "prompt_mask": prompt_mask,
    }
    with torch.no_grad():
        logits, scores = sam.predict_masks(
            pc_xyz, pc_rgb, prompt_points, prompt_labels, prompt_mask, prompt_mask is None
        )
    
    best_idx = torch.argmax(scores[0])
    prompt_mask = logits[0][best_idx][None, ...]
    
    global segment_mask
    segment_mask = return_mask = logits[0][best_idx] > 0
    
    return jsonify({"seg": return_mask.cpu().numpy().tolist()})


if __name__ == "__main__":
    app.run(host=f"{args.host}", port=f"{args.port}", debug=True)
