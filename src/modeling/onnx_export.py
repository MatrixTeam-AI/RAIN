# adapted from https://github.com/NVIDIA/TensorRT/blob/main/demo/Diffusion/utilities.py
#
# Copyright 2022 The HuggingFace Inc. team.
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import onnx
import gc
import onnx_graphsurgeon as gs
import torch
from onnx import shape_inference
from polygraphy.backend.onnx.loader import fold_constants

def export_onnx(
    model,
    onnx_path: str,
    opt_image_height: int,
    opt_image_width: int,
    opt_batch_size: int,
    onnx_opset: int,
    dtype,
    device,
    auto_cast: bool = True,
):
    from contextlib import contextmanager

    @contextmanager
    def auto_cast_manager(enabled):
        if enabled:
            with torch.inference_mode(), torch.autocast("cuda"):
                yield
        else:
            yield

    with auto_cast_manager(auto_cast):
        inputs = model.get_sample_input(opt_batch_size, opt_image_height, opt_image_width, dtype, device)
        torch.onnx.export(
            model,
            inputs,
            onnx_path,
            export_params=True,
            opset_version=onnx_opset,
            do_constant_folding=True,
            input_names=model.get_input_names(),
            output_names=model.get_output_names(),
            dynamic_axes=model.get_dynamic_axes(),
        )
    del model
    gc.collect()
    torch.cuda.empty_cache()


def optimize_onnx(
    onnx_path: str,
    onnx_opt_path: str,
):
    graph = gs.import_onnx(onnx.load(onnx_path))
    graph.cleanup().toposort()
    onnx_graph = fold_constants(gs.export_onnx(graph), allow_onnxruntime_shape_inference=True)
    onnx_graph = gs.export_onnx(graph)
    onnx.save_model(
        onnx_graph,
        onnx_opt_path,
        save_as_external_data=True,
        all_tensors_to_one_file=False,
        size_threshold=1024,
    )
    shape_inference.infer_shapes_path(onnx_opt_path, onnx_opt_path)
    graph = gs.import_onnx(onnx.load(onnx_opt_path))
    graph = gs.import_onnx(onnx_graph)
    graph.cleanup().toposort()
    onnx_opt_graph = gs.export_onnx(graph)
    onnx.save(
        onnx_opt_graph,
        onnx_opt_path,
        save_as_external_data=True,
        all_tensors_to_one_file=False,
        size_threshold=1024,
    )
    del onnx_opt_graph
    gc.collect()
    torch.cuda.empty_cache()


def handle_onnx_batch_norm(onnx_path: str):
    onnx_model = onnx.load(onnx_path)
    for node in onnx_model.graph.node:
        if node.op_type == "BatchNormalization":
            for attribute in node.attribute:
                if attribute.name == "training_mode":
                    if attribute.i == 1:
                        node.output.remove(node.output[1])
                        node.output.remove(node.output[1])
                    attribute.i = 0

    onnx.save_model(onnx_model, onnx_path)