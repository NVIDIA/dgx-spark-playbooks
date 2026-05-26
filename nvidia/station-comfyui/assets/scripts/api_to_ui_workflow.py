#!/usr/bin/env python3
"""
Convert ComfyUI API-format prompt graphs to UI workflow JSON (nodes + links).

API graphs are what /prompt expects inside {"prompt": ...}. The web UI "Load"
workflow expects this workflow shape (version 0.4).

Usage:
  python3 assets/scripts/api_to_ui_workflow.py path/to/workflow.api.json -o path/to/workflow.json
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

# Canonical ComfyUI input *slot* order (must match core node INPUT_TYPES / Schema order).
# When API JSON key order differs, links target the wrong slot without this map.
INPUT_ORDER: dict[str, list[str]] = {
    "CLIPTextEncode": ["text", "clip"],
    "VAEDecode": ["samples", "vae"],
    "SaveImage": ["images", "filename_prefix"],
    "VAELoader": ["vae_name"],
    "DualCLIPLoader": ["clip_name1", "clip_name2", "type", "device"],
    "QuadrupleCLIPLoader": ["clip_name1", "clip_name2", "clip_name3", "clip_name4"],
    "UNETLoader": ["unet_name", "weight_dtype"],
    "SamplerCustomAdvanced": ["noise", "guider", "sampler", "sigmas", "latent_image"],
    "KSamplerSelect": ["sampler_name"],
    "BasicScheduler": ["model", "scheduler", "steps", "denoise"],
    "BasicGuider": ["model", "conditioning"],
    "RandomNoise": ["noise_seed"],
    "FluxGuidance": ["conditioning", "guidance"],
    "EmptySD3LatentImage": ["width", "height", "batch_size"],
    "ModelSamplingFlux": ["model", "max_shift", "base_shift", "width", "height"],
    "KSampler": [
        "model",
        "seed",
        "steps",
        "cfg",
        "sampler_name",
        "scheduler",
        "positive",
        "negative",
        "latent_image",
        "denoise",
    ],
    "CLIPLoader": ["clip_name", "type", "device"],
    "EmptyHunyuanLatentVideo": ["width", "height", "length", "batch_size"],
    "ModelSamplingSD3": ["shift", "model"],
    "CLIPVisionLoader": ["clip_name"],
    "CLIPVisionEncode": ["clip_vision", "image", "crop"],
    # ComfyUI built-in order (see docs.comfy.org WanImageToVideo): links before size widgets.
    "WanImageToVideo": [
        "positive",
        "negative",
        "vae",
        "width",
        "height",
        "length",
        "batch_size",
        "clip_vision_output",
        "start_image",
    ],
    "LoadImage": ["image", "upload"],
    "CLIPTextEncodeHiDream": ["clip", "clip_l", "clip_g", "t5xxl", "llama"],
    "VAEDecodeTiled": ["samples", "vae", "tile_size", "overlap", "temporal_size", "temporal_overlap"],
    "AIO_Preprocessor": ["preprocessor", "resolution", "image"],
    "InstructPixToPixConditioning": ["positive", "negative", "vae", "pixels"],
    # docs.comfy.org: vae first, then spatial / temporal sizes, then optional images.
    "CosmosPredict2ImageToVideoLatent": [
        "vae",
        "width",
        "height",
        "length",
        "batch_size",
        "start_image",
        "end_image",
    ],
    "SaveAnimatedWEBP": ["images", "filename_prefix", "fps", "lossless", "quality", "method"],
}


def is_link(v: Any) -> bool:
    return isinstance(v, list) and len(v) == 2 and isinstance(v[0], (str, int))


def is_graph_node(key: str, node: Any) -> bool:
    if not isinstance(node, dict) or "class_type" not in node:
        return False
    try:
        int(key)
    except ValueError:
        return False
    return True


def iter_graph(api: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return [(k, v) for k, v in api.items() if is_graph_node(k, v)]


def ordered_input_items(class_type: str, inputs: dict[str, Any]) -> list[tuple[str, Any]]:
    order = INPUT_ORDER.get(class_type)
    if not order:
        return list(inputs.items())
    seen: set[str] = set()
    out: list[tuple[str, Any]] = []
    for k in order:
        if k in inputs:
            out.append((k, inputs[k]))
            seen.add(k)
    for k, v in inputs.items():
        if k not in seen:
            out.append((k, v))
    return out


def collect_links(api: dict[str, Any]) -> list[tuple[int, int, int, int, int, str]]:
    links: list[tuple[int, int, int, int, int, str]] = []
    link_id = 1
    for dst_key, node in iter_graph(api):
        dst_id = int(dst_key)
        inputs = node.get("inputs") or {}
        for slot_idx, (iname, val) in enumerate(ordered_input_items(node["class_type"], inputs)):
            if not is_link(val):
                continue
            src_id = int(val[0])
            src_slot = int(val[1])
            links.append((link_id, src_id, src_slot, dst_id, slot_idx, "*"))
            link_id += 1
    return links


def link_for_target(
    links: list[tuple[int, int, int, int, int, str]], dst_id: int, input_slot_index: int
) -> int | None:
    for lid, src, ss, did, ds, _ in links:
        if did == dst_id and ds == input_slot_index:
            return lid
    return None


def build_ui_workflow(api: dict[str, Any]) -> dict[str, Any]:
    links_raw = collect_links(api)
    last_link_id = max((l[0] for l in links_raw), default=0)

    nodes_out: list[dict[str, Any]] = []
    node_ids = sorted(int(k) for k, _ in iter_graph(api))
    last_node_id = max(node_ids) if node_ids else 0

    for order, nid in enumerate(node_ids):
        key = str(nid)
        node = api[key]
        ctype = node["class_type"]
        inputs = node.get("inputs") or {}
        input_items = ordered_input_items(ctype, inputs)

        ui_inputs: list[dict[str, Any]] = []
        widgets_values: list[Any] = []

        for slot_idx, (iname, val) in enumerate(input_items):
            lid = link_for_target(links_raw, nid, slot_idx)
            if is_link(val):
                ui_inputs.append({"name": iname, "type": "*", "link": lid})
            else:
                ui_inputs.append({"name": iname, "type": "*", "link": None})
                widgets_values.append(val)

        out_links: dict[int, list[int]] = {}
        max_slot = 0
        for lid, src, ss, _, _, _ in links_raw:
            if src == nid:
                out_links.setdefault(ss, []).append(lid)
                max_slot = max(max_slot, ss)

        ui_outputs: list[dict[str, Any]] = []
        for slot in range(max_slot + 1):
            ls = out_links.get(slot, [])
            ui_outputs.append(
                {"name": str(slot), "type": "*", "links": ls, "slot_index": slot}
            )

        col = order % 5
        row = order // 5
        pos = [80 + col * 420, 40 + row * 220]

        nodes_out.append(
            {
                "id": nid,
                "type": ctype,
                "pos": pos,
                "size": [320, 120],
                "flags": {},
                "order": order,
                "mode": 0,
                "inputs": ui_inputs,
                "outputs": ui_outputs,
                "properties": {
                    "cnr_id": "comfy-core",
                    "ver": "0.3.0",
                    "Node name for S&R": ctype,
                },
                "widgets_values": widgets_values,
            }
        )

    links_arr: list[list[Any]] = [
        [lid, src, ss, did, ds, typ] for lid, src, ss, did, ds, typ in links_raw
    ]

    return {
        "last_node_id": last_node_id,
        "last_link_id": last_link_id,
        "nodes": nodes_out,
        "links": links_arr,
        "groups": [],
        "config": {},
        "extra": {"ds": {"scale": 1, "offset": [0, 0]}},
        "version": 0.4,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_api_json")
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    with open(args.input_api_json, encoding="utf-8") as f:
        api = json.load(f)

    if not isinstance(api, dict):
        print("Input must be a JSON object (API prompt graph)", file=sys.stderr)
        sys.exit(1)

    ui = build_ui_workflow(api)
    text = json.dumps(ui, indent=2)

    if args.output == "-":
        print(text)
    else:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    main()
