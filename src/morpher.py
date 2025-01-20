from omegaconf import OmegaConf
import os
import torch
import numpy as np
from PIL import Image
import time
import gc
import cv2
from src.modeling.translation import translation
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.pose_guider import PoseGuider
from src.taesdv.taesdv import TAESDV
from src.models.unet_3d import UNet3DConditionModel
from src.models.unet_3d_explicit_reference import UNet3DConditionModelExplicitReference
from src.models.mutual_self_attention import ReferenceAttentionControl
from src.scheduler.scheduler_lcm import LCMScheduler
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
import onnxruntime as ort
from src.modeling.engine_model import EngineModel
from collections import deque
from threading import Lock, Thread
from torchvision import transforms
from einops import rearrange




def map_device(device_or_str):
    return device_or_str if isinstance(device_or_str, torch.device) else torch.device(device_or_str)

class RAINMorpher:
    def __init__(self, config_path, device=None):
        cfg = OmegaConf.load(config_path)
        if(device is None):
            self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = map_device(device)

        device_id = None
        if(self.device.type == "cuda"):
            device_id = self.device.index
        
        self.device_id = device_id
        
        self.temporal_adaptive_step = cfg.temporal_adaptive_step
        self.temporal_window_size = cfg.temporal_window_size
        if cfg.dtype == "fp16":
            self.numpy_dtype = np.float16
            self.dtype = torch.float16
        elif cfg.dtype == "fp32":
            self.numpy_dtype = np.float32
            self.dtype = torch.float32
        
        self.tensorrt = cfg.tensorrt

        sched_kwargs = OmegaConf.to_container(cfg.noise_scheduler_kwargs)

        providers = [('CUDAExecutionProvider', {'device_id': device_id,})] if self.device.type == "cuda" else None
        if cfg.onnx_tensorrt:
            providers = [('TensorrtExecutionProvider', {'device_id': device_id,})] + providers

        self.providers = providers

        def guidance_scale_embedding(w, embedding_dim=512, dtype=torch.float32):
            assert len(w.shape) == 1
            w = w * 1000.0
            half_dim = embedding_dim // 2
            emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
            emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
            emb = w.to(dtype)[:, None] * emb[None, :]
            emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
            if embedding_dim % 2 == 1:  # zero pad
                emb = torch.nn.functional.pad(emb, (0, 1))
            assert emb.shape == (w.shape[0], embedding_dim)
            return emb

        w = cfg.guidance_scale
        w = torch.tensor([w])
        w_embedding = guidance_scale_embedding(w, embedding_dim=cfg.unet_additional_kwargs.cond_proj_dim)
        self.w_embedding = w_embedding.to(device=self.device, dtype=self.dtype)
        self.num_inference_steps = cfg.num_inference_steps

        if self.tensorrt:
            from src.modeling.framed_models import unet_work, reference_net, vae_encoder, clip_model
            from src.modeling.onnx_export import (
                export_onnx,
                handle_onnx_batch_norm,
                optimize_onnx,
            )
            from polygraphy.backend.trt import engine_from_network, network_from_onnx_path, save_engine, CreateConfig
            from polygraphy.logger import G_LOGGER
            import tensorrt as trt
            G_LOGGER.severity = G_LOGGER.VERBOSE

            if(not os.path.exists(cfg.vae_encoder_onnx_path)):
                print("Exporting VAE Encoder Onnx Model...")
                if(not os.path.exists(os.path.dirname(cfg.vae_encoder_onnx_path))):
                    os.mkdir(os.path.dirname(cfg.vae_encoder_onnx_path))
                vae = AutoencoderKL.from_pretrained(cfg.vae_model_path).to(
                    device=self.device, dtype=self.dtype
                )
                model = vae_encoder(vae)
                export_onnx(
                    model,
                    onnx_path=cfg.vae_encoder_onnx_path,
                    opt_image_height=cfg.height,
                    opt_image_width=cfg.width,
                    opt_batch_size=cfg.batch_size,
                    onnx_opset=cfg.onnx_opset,
                    auto_cast=True,
                    dtype=self.dtype,
                    device=self.device,
                )
                del model
                gc.collect()
                torch.cuda.empty_cache()
            
            self.vae = ort.InferenceSession(cfg.vae_encoder_onnx_path, providers=providers)

            if(not os.path.exists(cfg.clip_onnx_path)):
                print("Exporting CLIP Encoder Onnx Model...")
                if(not os.path.exists(os.path.dirname(cfg.clip_onnx_path))):
                    os.mkdir(os.path.dirname(cfg.clip_onnx_path))
                image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                    cfg.image_encoder_path,
                ).to(dtype=torch.float16, device="cuda")
                model = clip_model(image_encoder)
                export_onnx(
                    model,
                    onnx_path=cfg.clip_onnx_path,
                    opt_image_height=cfg.height,
                    opt_image_width=cfg.width,
                    opt_batch_size=cfg.batch_size,
                    onnx_opset=cfg.onnx_opset,
                    auto_cast=True,
                    dtype=self.dtype,
                    device=self.device,
                )
                del model
                gc.collect()
                torch.cuda.empty_cache()
            
            self.image_encoder = ort.InferenceSession(cfg.clip_onnx_path, providers=providers)

            if(not os.path.exists(cfg.reference_net_onnx_path)):
                print("Exporting Reference UNet Onnx Model...")
                if(not os.path.exists(os.path.dirname(cfg.reference_net_onnx_path))):
                    os.mkdir(os.path.dirname(cfg.reference_net_onnx_path))
                reference_unet = UNet2DConditionModel.from_config(
                    cfg.unet_config_path,
                ).to("cuda",dtype=torch.float16)
                reference_control_writer = ReferenceAttentionControl(
                    reference_unet,
                    do_classifier_free_guidance=False,
                    mode="write",
                    batch_size=cfg.batch_size,
                    fusion_blocks="full",
                    cache_kv=False,
                )
                reference_unet_state_dict = torch.load(cfg.reference_unet_weight_path, map_location="cpu")
                reference_unet.load_state_dict(reference_unet_state_dict)
                del reference_unet_state_dict
                model = reference_net(reference_unet, reference_control_writer)
                export_onnx(
                    model,
                    onnx_path=cfg.reference_net_onnx_path,
                    opt_image_height=cfg.height,
                    opt_image_width=cfg.width,
                    opt_batch_size=cfg.batch_size,
                    onnx_opset=cfg.onnx_opset,
                    auto_cast=True,
                    dtype=self.dtype,
                    device=self.device,
                )
                del model
                gc.collect()
                torch.cuda.empty_cache()
            
            self.reference_unet = ort.InferenceSession(cfg.reference_net_onnx_path, providers=providers)

            if(not os.path.exists(cfg.tensorrt_target_model)):
                print("Building TensorRT Engine...")
                if(not os.path.exists(cfg.onnx_opt_path)):
                    if(not os.path.exists(cfg.onnx_path)):
                        print("Exportng Onnx Model...")
                        pose_guider = PoseGuider(
                            conditioning_embedding_channels=320, block_out_channels=(16, 32, 96, 256)
                        ).to(device=self.device, dtype=self.dtype)
                        pose_guider_state_dict = torch.load(cfg.pose_guider_weight_path, map_location="cpu")
                        pose_guider.load_state_dict(pose_guider_state_dict)
                        del pose_guider_state_dict
                        unet = UNet3DConditionModelExplicitReference.from_pretrained_2d(
                            cfg.denoising_unet_weight_path,
                            unet_additional_kwargs=OmegaConf.to_container(
                                cfg.unet_additional_kwargs
                            ),
                            direct_weight=True,
                            config_path=cfg.unet_config_path
                        ).to(device=self.device, dtype=self.dtype)
                        vae = TAESDV(cfg.taesdv_weight_path)
                        vae.to(device=self.device, dtype=self.dtype)
                        scheduler = LCMScheduler(**sched_kwargs)
                        scheduler.to(device)
                        scheduler.set_timesteps(4)
                        timestep = torch.tensor([[249, 249, 249, 249, 499, 499, 499, 499, 749, 749, 749, 749, 999, 999, 999, 999]],dtype=torch.long,device=self.device)
                        model = unet_work(pose_guider, unet, vae, scheduler, timestep, self.w_embedding)
                        if(not os.path.exists(os.path.dirname(cfg.onnx_path))):
                            os.mkdir(os.path.dirname(cfg.onnx_path))
                        export_onnx(
                            model,
                            onnx_path=cfg.onnx_path,
                            opt_image_height=cfg.height,
                            opt_image_width=cfg.width,
                            opt_batch_size=cfg.batch_size,
                            onnx_opset=cfg.onnx_opset,
                            auto_cast=True,
                            dtype=self.dtype,
                            device=self.device,
                        )
                        del model
                        gc.collect()
                        torch.cuda.empty_cache()
                        
                    print("Optimizing Onnx Model...")
                    if(not os.path.exists(os.path.dirname(cfg.onnx_opt_path))):
                        os.mkdir(os.path.dirname(cfg.onnx_opt_path))
                    optimize_onnx(
                        onnx_path=cfg.onnx_path,
                        onnx_opt_path=cfg.onnx_opt_path,
                    )
                gc.collect()
                torch.cuda.empty_cache()
                engine = engine_from_network(
                    network_from_onnx_path(cfg.onnx_opt_path, flags=[trt.OnnxParserFlag.NATIVE_INSTANCENORM]),
                    config=CreateConfig(
                        fp16=True, refittable=False, 
                    ),
                )
                save_engine(engine, path=cfg.tensorrt_target_model)
                gc.collect()
                torch.cuda.empty_cache()
            
            self.unet_work = None

        else:
            self.pose_guider = PoseGuider(
                conditioning_embedding_channels=320, block_out_channels=(16, 32, 96, 256)
            ).to(device=self.device, dtype=self.dtype)
            pose_guider_state_dict = torch.load(cfg.pose_guider_weight_path, map_location="cpu")
            self.pose_guider.load_state_dict(pose_guider_state_dict)
            del pose_guider_state_dict

            self.denoising_unet = UNet3DConditionModel.from_pretrained_2d(
                cfg.denoising_unet_weight_path,
                unet_additional_kwargs=OmegaConf.to_container(
                    cfg.unet_additional_kwargs
                ),
                direct_weight=True,
                config_path=cfg.unet_config_path
            ).to(device=self.device, dtype=self.dtype)

            self.reference_unet = UNet2DConditionModel.from_config(
                cfg.unet_config_path,
            ).to(device=self.device, dtype=self.dtype)
            reference_unet_state_dict = torch.load(cfg.reference_unet_weight_path, map_location="cpu")
            self.reference_unet.load_state_dict(reference_unet_state_dict)
            del reference_unet_state_dict

            self.reference_control_writer = ReferenceAttentionControl(
                self.reference_unet,
                do_classifier_free_guidance=False,
                mode="write",
                batch_size=cfg.batch_size,
                fusion_blocks="full",
            )
            self.reference_control_reader = ReferenceAttentionControl(
                self.denoising_unet,
                do_classifier_free_guidance=False,
                mode="read",
                batch_size=cfg.batch_size,
                fusion_blocks="full",
            )
            self.vae = AutoencoderKL.from_pretrained(cfg.vae_model_path).to(
                device=self.device, dtype=self.dtype
            )
            self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                cfg.image_encoder_path,
            ).to(device=self.device, dtype=self.dtype)
            self.latents_pile = deque([])
            self.pose_pile = deque([])
            self.latents_pile_fullness = False
        
        self.scheduler = LCMScheduler(**sched_kwargs)
        self.scheduler.to(self.device)
        self.scheduler.set_timesteps(self.num_inference_steps)

        patterns = [
            ([1,2], (0, 255, 255)),
            ([4], (0, 255, 255)),
            ([8], (0, 255, 255)),
            ([12], (0, 255, 255)),
            ([15, 14], (0, 255, 255)),
            ([17], (128, 0, 255)),
            ([18, 19], (128, 0, 255)),
            ([20], (128, 0, 255)),
            ([23], (255, 0, 128)),
            ([24, 25], (255, 0, 128)),
            ([26], (255, 0, 128)),
            ([60], (0, 0, 255)),
            ([62], (0, 0, 255)),
            ([64], (0, 0, 255)),
            ([66], (0, 0, 255)),
            ([33, 30], (255, 0, 255)),
            ([36, 37], (255, 128, 128)),
            ([37, 38], (255, 128, 128)),
            ([38, 39], (255, 128, 128)),
            ([40, 39], (255, 128, 128)),
            ([41], (255, 128, 128)),
            ([42, 43], (128, 128, 255)),
            ([43, 44], (128, 128, 255)),
            ([44, 45], (128, 128, 255)),
            ([46], (128, 128, 255)),
            ([47, 42], (128, 128, 255)),
        ]
        
        """if(cfg.landmark_detector == "pfld"):
            from src.modeling.pfldlandmark import PFLDFaceTracker
            translations = [
                translation(0, 68, vertical_scale=0.75, xaxis=(2, 14)),
                translation(36, 42,vertical_scale=2.4, xaxis=(0,3), horizontal_scale=1.4, vertical_transition=15, horizontal_transition=-5),
                translation(42, 48,vertical_scale=2.4, xaxis=(0,3), horizontal_scale=1.4, vertical_transition=15, horizontal_transition=5),   
                translation(60, 68,vertical_scale=1.2, xaxis=(0, 4), horizontal_scale=0.67, vertical_transition=0.00),
                translation(36, 42, xaxis=(0,3), relative_length=(33, 13), horizontal_transition=0.2),
                translation(42, 48, xaxis=(0,3), relative_length=(33, 13), horizontal_transition=0.2),
                translation(36, 42, xaxis=(0,3), relative_length=(33, 3), horizontal_transition=-0.2),
                translation(42, 48, xaxis=(0,3), relative_length=(33, 3), horizontal_transition=-0.2),
                translation(60, 68, xaxis=(0,4), relative_length=(33, 13), horizontal_transition=0.2),
                translation(60, 68, xaxis=(0,4), relative_length=(33, 3), horizontal_transition=-0.15),
                translation(4, 13, xaxis=(0,8), horizontal_scale=0.9),
            ]
            self.face_tracker = PFLDFaceTracker(cfg.pfld_detection_model_path, cfg.pfld_landmark_model_path, translations=translations, patterns=patterns, threshold=cfg.threshold, providers=providers)
        el"""
        if(cfg.landmark_detector == "dwpose"):
            from src.dwpose import DWposeDetector
            translations = ([
                translation(0, 68, vertical_scale=0.75, xaxis=(2, 14)),
                translation(36, 42,vertical_scale=3.0, xaxis=(0,3), horizontal_scale=1.6,),
                translation(42, 48,vertical_scale=3.0, xaxis=(0,3), horizontal_scale=1.6),
                translation(60, 68,vertical_scale=1.2, xaxis=(0, 4), horizontal_scale=0.67),
                translation(36, 42, xaxis=(0,3), relative_length=(33, 13), horizontal_transition=0.10),
                translation(42, 48, xaxis=(0,3), relative_length=(33, 13), horizontal_transition=0.15),
                translation(36, 42, xaxis=(0,3), relative_length=(33, 3), horizontal_transition=-0.15),
                translation(42, 48, xaxis=(0,3), relative_length=(33, 3), horizontal_transition=-0.10),
                translation(36, 42, xaxis=(0,3), relative_length=(19, 6), vertical_transition=0.10),
                translation(42, 48, xaxis=(0,3), relative_length=(26, 12), vertical_transition=0.10),
                translation(60, 68, xaxis=(0,4), relative_length=(33, 13), horizontal_transition=0.15),
                translation(60, 68, xaxis=(0,4), relative_length=(33, 3), horizontal_transition=-0.15),
                translation(4, 13, xaxis=(0,8), horizontal_scale=0.9),
            ])
            self.face_tracker = DWposeDetector(cfg.dwpose_detection_model_path, cfg.dwpose_landmark_model_path, providers, face_only=True, translations=translations, detection_size=(512, 512))
            self.face_tracker.set_patterns(patterns)
        else:
            raise NotImplementedError
        

        self.generator = torch.Generator(self.device)
        self.generator.manual_seed(cfg.seed)

        self.batch_size = cfg.batch_size
        self.vae_scale_factor = 8
        self.ref_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True
        )
        self.clip_image_processor = CLIPImageProcessor()

        self.encoder_hidden_states = None

        self.latent_shape = (
            self.batch_size,
            4,
            self.temporal_adaptive_step,
            cfg.height // self.vae_scale_factor,
            cfg.width // self.vae_scale_factor,
        )

        
            
        self.cfg = cfg
        torch.cuda.empty_cache()

        self.input_pile = deque([])
        self.input_pile_lock = Lock()
        self.output_pile = deque([])
        self.output_pile_lock = Lock()
        self.pose_transform = transforms.Compose(
            [transforms.Resize((self.cfg.height, self.cfg.width))]
        )

        self.fps = cfg.fps
        self.timestamp = -1
        self.work_thread = None
        self.work_flag = False
        self.reference_hidden_states_names = ["d00", "d01", "d10", "d11", "d20", "d21", "m", "u10", "u11", "u12", "u20", "u21", "u22", "u30", "u31", "u32"]
        self.encoder_hidden_states = None
        self.reference_hidden_states = None


    def seed(self, x):
        self.generator.manual_seed(x)

    def enable_xformers_memory_efficient_attention(self):
        self.reference_unet.enable_xformers_memory_efficient_attention()
        if not self.tensorrt:
            self.denoising_unet.enable_xformers_memory_efficient_attention()
        
    def load_reference_part_to_cpu_or_uninstall(self):
        if not self.tensorrt:
            self.image_encoder.to(torch.device("cpu"))
            self.reference_unet.to(torch.device("cpu"))
        else:
            del self.image_encoder
            del self.reference_unet
            del self.vae
            self.image_encoder = None
            self.reference_unet = None
            self.vae = None

        torch.cuda.empty_cache()

    def load_reference_part_to_device(self, device=None):
        if(device is None):
            device = self.device
        if not self.tensorrt:
            self.image_encoder.to(device)
            self.reference_unet.to(device)
        else:
            self.vae = ort.InferenceSession(self.cfg.vae_encoder_onnx_path, providers=self.providers)
            self.reference_unet = ort.InferenceSession(self.cfg.reference_net_onnx_path, providers=self.providers)
            self.image_encoder = ort.InferenceSession(self.cfg.clip_onnx_path, providers=self.providers)


    def clear_reference(self):
        if not self.tensorrt:
            self.reference_control_reader.clear()
            self.reference_control_writer.clear()
        self.encoder_hidden_states = None

    def clear_queue(self):
        self.input_pile_lock.acquire()
        self.input_pile = deque([])
        self.input_pile_lock.release()
        if not self.tensorrt:
            self.latents_pile = deque([])
            self.pose_pile = deque([])
            self.latents_pile_fullness = False
        self.output_pile_lock.acquire()
        self.output_pile = deque([])
        self.output_pile_lock.release()

    @torch.no_grad()
    def fuse_reference(self, ref_image): # pil input
        if(ref_image.__class__.__name__ == "ndarray"):
            ref_image = Image.fromarray(ref_image)
        clip_image = self.clip_image_processor.preprocess(
            ref_image, return_tensors="pt"
        ).pixel_values
        ref_image_tensor = self.ref_image_processor.preprocess(
            ref_image, height=self.cfg.reference_image_height, width=self.cfg.reference_image_width
        )  # (bs, c, width, height)
        if self.tensorrt:
            clip_image_embeds = self.image_encoder.run(None, {"images" : clip_image.to(dtype=torch.float16).numpy()})[0]
            encoder_hidden_states = clip_image_embeds[:, None, :]
            ref_image_latents = self.vae.run(None, {"images" : ref_image_tensor.to(dtype=torch.float16).numpy()})[0]
            reference_hidden_states = self.reference_unet.run(None, {
                "latents" : ref_image_latents,
                "clip_embeds" : encoder_hidden_states
            })
            self.encoder_hidden_states = encoder_hidden_states
            self.reference_hidden_states = reference_hidden_states
            if not(self.unet_work is None):
                self.unet_work.prefill(encoder_hidden_states = encoder_hidden_states)
                self.unet_work.prefill(**{self.reference_hidden_states_names[i]: reference_hidden_states[i] for i in range(len(self.reference_hidden_states_names))})

        else:
            clip_image_embeds = self.image_encoder(
                clip_image.to(self.image_encoder.device, dtype=self.image_encoder.dtype)
            ).image_embeds
            self.encoder_hidden_states = clip_image_embeds.unsqueeze(1)
            ref_image_tensor = ref_image_tensor.to(
                dtype=self.vae.dtype, device=self.vae.device
            )
            ref_image_latents = self.vae.encode(ref_image_tensor).latent_dist.mean
            ref_image_latents = ref_image_latents * 0.18215  # (b, 4, h, w)
            self.reference_unet(
                ref_image_latents.to(self.reference_unet.device),
                torch.zeros((self.batch_size,),dtype=self.dtype,device=self.reference_unet.device),
                encoder_hidden_states=self.encoder_hidden_states,
                return_dict=False,
            )
            self.reference_control_reader.update(self.reference_control_writer)
            self.encoder_hidden_states = self.encoder_hidden_states.to(self.device)

    @torch.no_grad()
    def process_input(self, wait_after=0.0):
        time.sleep(wait_after)
        self.input_pile_lock.acquire()
        if(len(self.input_pile) < self.cfg.temporal_window_size):
            self.input_pile_lock.release()
            return
        poses = []
        for i in range(self.cfg.temporal_window_size):
            poses.append(self.detect(self.input_pile.popleft()))
        self.input_pile_lock.release()
        input_poses = torch.cat([
                torch.from_numpy(self.numpy_dtype(poses[i])).to(self.device)
                .permute(2, 0, 1).unsqueeze(1) / 255.0
                for i in range(self.temporal_window_size)
                ], dim=1)
        pose_cond_tensor = input_poses.unsqueeze(0)
        if self.tensorrt:
            if(self.unet_work is None):
                return
            latents = torch.randn(
                    self.latent_shape, generator=self.generator, device=self.device, dtype=self.dtype
                ) * self.scheduler.init_noise_sigma
            lcm_noise = torch.randn(
                tuple(self.unet_work.input_shapes["add_noise"]), generator=self.generator, device=self.device, dtype=self.dtype
            ) 
            images = self.unet_work(output_list=["image"], return_tensor=True, pose=pose_cond_tensor, add_noise=lcm_noise, new_noise=latents)["image"][0]
            images = images.permute(0,2,3,1).cpu().numpy()
            self.output_pile_lock.acquire()
            for i in range(images.shape[0]):
                self.output_pile.append(images[i])
            self.output_pile_lock.release()
        else:
            pose_cond_tensor = pose_cond_tensor.to(
                device=self.device, dtype=self.dtype
            )
            pose_fea = self.pose_guider(pose_cond_tensor)
            self.pose_pile.append(pose_fea)
            latents = torch.randn(
                self.latent_shape, generator=self.generator, device=self.device, dtype=self.dtype
            ) * self.scheduler.init_noise_sigma
            self.latents_pile.append(latents)

            while(len(self.pose_pile) > self.temporal_window_size):
                self.pose_pile.popleft()

            while(len(self.latents_pile) > self.temporal_window_size):
                self.latents_pile.popleft()

            step_length = self.scheduler.config.num_train_timesteps // self.num_inference_steps
            jump = self.num_inference_steps // self.temporal_window_size
            
            if len(self.latents_pile) == self.temporal_window_size:
                self.latents_pile_fullness = True
                l, r = 0, self.temporal_window_size
            else:
                if(self.latents_pile_fullness):
                    l, r = 0, len(self.latents_pile)
                else:
                    l, r = - len(self.latents_pile), self.temporal_window_size
            ut = self.scheduler.timesteps[
                (torch.linspace(self.cfg.num_inference_steps, 0, self.temporal_window_size + 1)[1:].long())
            ] 
            latents_model_input = torch.cat(list(self.latents_pile), dim=2)
            for j in range(jump):
                timesteps = ut[l : r] - step_length * j
                timesteps = torch.stack([torch.stack([timesteps] * self.temporal_adaptive_step).T.reshape((-1))] * self.batch_size).to(self.device)
                noise_pred = self.denoising_unet(
                    latents_model_input,
                    timesteps,
                    encoder_hidden_states=self.encoder_hidden_states,
                    pose_cond_fea=torch.cat(list(self.pose_pile), dim=2),
                    timestep_cond=self.w_embedding,
                    return_dict=False,
                )[0]
                latents_model_input = self.scheduler.step(
                    noise_pred, timesteps, latents_model_input, generator=self.generator, return_dict=False
                )[0]
                latents_model_input = latents_model_input.to(dtype=self.dtype)

            for i in range(len(self.latents_pile)):
                self.latents_pile[i] = latents_model_input[:, :, i * self.temporal_adaptive_step : (i + 1) * self.temporal_adaptive_step, :, :]
        
            if(self.latents_pile_fullness):
                latents = self.latents_pile.popleft()
                latents = 1 / 0.18215 * latents
                fdim = latents.shape[2]
                latents = rearrange(latents, "b c f h w -> (b f) c h w")
                video = self.vae.decode(latents).sample
                video = rearrange(video, "(b f) c h w -> b f h w c", f=fdim)
                video = (video / 2 + 0.5).clamp(0, 1)
                video = video[0].cpu().numpy()
                self.output_pile_lock.acquire()
                for i in range(video.shape[0]):
                    self.output_pile.append(video[i])
                self.output_pile_lock.release()
                
                
        
    def detect(self, face_image):
        return self.face_tracker.detect(face_image)
    
    def fetch_one_frame(self, wait_after=0.0):
        time.sleep(wait_after)
        self.output_pile_lock.acquire()
        if(len(self.output_pile) == 0):
            self.output_pile_lock.release()
            return None
        output = self.output_pile.popleft()
        self.output_pile_lock.release()
        return output
    
    def set_translations(self, translations):
        self.face_tracker.set_translations(translations)

    def set_patterns(self, patterns):
        self.face_tracker.set_patterns(patterns)

    def set_fps(self, fps):
        self.fps = fps

    def push_input(self, x, mix=1.0, eager_mode = False, buffer_limit=6): # numpy or pil input
        if(eager_mode):
            self.input_pile_lock.acquire()
            self.input_pile.append(np.array(x))
            self.input_pile_lock.release()
            return
        
        current_timestamp = time.time()
        if(current_timestamp - self.timestamp >= 1.0 / self.fps and len(self.input_pile) <= buffer_limit):
            self.input_pile_lock.acquire()
            self.input_pile.append(np.array(x))
            self.input_pile_lock.release()
            self.timestamp = current_timestamp * mix + (self.timestamp + 1.0 / self.fps) * (1.0 - mix)
        else:
            pass

    def kick_start(self, wait_after=0.01):
        if (self.work_flag or not (self.work_thread is None)):
            return
        else:
            self.work_flag = True

        def thread_func(self_, wait_after):
            if(self_.tensorrt):
                self_.unet_work = EngineModel(engine_file_path=self_.cfg.tensorrt_target_model, device_int=self_.device_id)
                self_.unet_work.bind({
                    "pose_cond_fea_out" : "pose_cond_fea",
                    "latents" : "sample",
                    "omem0" : "pmem0",
                    "omem1" : "pmem1",
                    "omem2" : "pmem2",
                    "omem3" : "pmem3",
                    "omem4" : "pmem4",
                    "omem5" : "pmem5",
                    "omem6" : "pmem6",
                    "omem7" : "pmem7",
                    "omem8" : "pmem8",
                    "omem9" : "pmem9",
                })
                if not (self_.encoder_hidden_states is None):
                    self_.unet_work.prefill(encoder_hidden_states = self_.encoder_hidden_states)
                if not (self_.reference_hidden_states is None):
                    self_.unet_work.prefill(**{self_.reference_hidden_states_names[i]: self_.reference_hidden_states[i] for i in range(len(self_.reference_hidden_states_names))})

            
            while self_.work_flag:
                self_.process_input(wait_after=wait_after)

            if self_.tensorrt:
                del self_.unet_work
                self_.unet_work = None
                gc.collect()
            

        self.work_thread = Thread(target=thread_func, args=(self, wait_after))
        self.work_thread.start()

    def stop(self):
        self.work_flag = False
        if not (self.work_thread is None):
            self.work_thread.join()
            self.work_thread = None

    def videomxn(self, vids, m, n=None, size=None, length=None, centercrop=True):
        if length is None:
            length = min([len(vid) for vid in vids])
        
        if size is None:
            size = vids[0][0].shape[:2]

        if n is None:
            n = len(vids) // m

        rvid = []

        for i in range(length):
            w = np.zeros((size[0] * m, size[1] * n, 3), dtype=np.uint8)
            for j in range(m):
                for k in range(n):
                    w[size[0] * j:size[0] * (j + 1), size[1] * k:size[1] * (k + 1), :] = self.verify(vids[j*n + k][i], size, centercrop)
            rvid.append(w)
        return rvid

    def save_video(self, imgs, vid_file_name, fps=24):
        video_codec = cv2.VideoWriter_fourcc(*"mp4v")
        segment_file = cv2.VideoWriter(vid_file_name, video_codec, fps, (imgs[0].shape[1], imgs[0].shape[0]))
        if(imgs[0].dtype == np.uint8):
            for f in imgs:
                segment_file.write(f[:,:,::-1])
        else:
            for f in imgs:
                segment_file.write(np.uint8(255*f)[:,:,::-1])
        
        segment_file.release()


    def verify(self, img, size=None, centercrop=False):
        input = img
        if isinstance(input, Image.Image):
            input = np.array(input)
        if input.dtype != np.uint8:
            input = np.uint8(255 * input)
        if not(size is None) and input.shape[:2] != size:
            if centercrop:
                hs, ws = input.shape[:2]
                ht, wt = size
                hws = hs / ws
                hwt = ht / wt
                if hws > hwt:
                    mw = ws
                    mh = int(ws * ht / wt)
                else:
                    mh = hs
                    mw = int(hs * wt / ht)
                dw = max((ws - mw) // 2, 1)
                dh = max((hs - mh) // 2, 1)

                input = np.array(Image.fromarray(input[dh:-dh,dw:-dw]).resize((size[1], size[0])))
            else:
                input = np.array(Image.fromarray(input).resize((size[1], size[0])))

        return input

    def imgmxn(self, imgs, m, n=None, size = None, centercrop=False):
        if n is None:
            n = len(imgs) // m
        if size is None:
            size = self.verify(imgs[0]).shape[:2]
        w = np.zeros((size[0] * m, size[1] * n, 3), dtype=np.uint8)
        for i in range(m):
            for j in range(n):
                w[size[0] * i: size[0] * (i + 1), size[1] * j:size[1] * (j + 1),:] = self.verify(imgs[i * n + j], size=size, centercrop=centercrop)

        return w
    
    @torch.no_grad()
    def latentsmxn(self, latents, m, n=None, size=None, centercrop=False):
        if size is None:
            size = (latents[0].shape[-2] * 8, latents[0].shape[-1] * 8)
        if self.tensorrt:
            return None
        w = np.zeros((size[0] * m, size[1] * n, 3), dtype=np.uint8)
        f = latents[0].shape[2]
        for i in range(m):
            for j in range(n):
                w[size[0] * i:size[0] * (i + 1), size[1] * j:size[1] * (j + 1),:] = self.verify(np.uint8(255 * self.vae.decode(1 / 0.18215 * latents[(i * n + j) // f][:,:,(i * n + j) % f,:,:]).sample[0].permute(1,2,0).clamp(0,1).cpu().numpy()), size=size, centercrop=centercrop)
        return w