import inspect
from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import numpy as np
import torch
from diffusers import DiffusionPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers import (DDIMScheduler, DPMSolverMultistepScheduler,
                                  EulerAncestralDiscreteScheduler,
                                  EulerDiscreteScheduler, LMSDiscreteScheduler,
                                  PNDMScheduler)
from diffusers.utils import BaseOutput, is_accelerate_available
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from tqdm import tqdm
from transformers import CLIPImageProcessor

from src.models.mutual_self_attention import ReferenceAttentionControl


@dataclass
class Pose2VideoPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]


class Pose2VideoPipelineLCM(DiffusionPipeline):
    _optional_components = []

    def __init__(
        self,
        vae,
        image_encoder,
        reference_unet,
        denoising_unet,
        pose_guider,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
        image_proj_model=None,
        tokenizer=None,
        text_encoder=None,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            image_encoder=image_encoder,
            reference_unet=reference_unet,
            denoising_unet=denoising_unet,
            pose_guider=pose_guider,
            scheduler=scheduler,
            image_proj_model=image_proj_model,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.clip_image_processor = CLIPImageProcessor()
        self.ref_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True
        )
        self.cond_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,
            do_convert_rgb=True,
            do_normalize=False,
        )

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def decode_latents(self, latents, vae_bsz, hiddens=None, push_ready=False):

        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        img_bsz = latents.shape[0]
        # video = self.vae.decode(latents).sample
        video = []
        if(hiddens is None):
            for frame_idx in tqdm(range((img_bsz + vae_bsz - 1) // vae_bsz)):
                video.append(self.vae.decode(latents[frame_idx * vae_bsz : min((frame_idx + 1) * vae_bsz, img_bsz)]).sample)
        else:
            hiddens = [hidden.repeat(vae_bsz, 1, 1, 1) for hidden in hiddens]
            for frame_idx in tqdm(range((img_bsz + vae_bsz - 1) // vae_bsz)):
                if(push_ready and frame_idx > 0):
                    p, hiddens = self.vae.encode(video[-1:], return_dict=False , return_hiddens=True) 
                    hiddens = [hidden.repeat(vae_bsz, 1, 1, 1) for hidden in hiddens]
                video.append(self.vae.decode(latents[frame_idx * vae_bsz : min((frame_idx + 1) * vae_bsz, img_bsz)], hiddens=hiddens).sample)
        video = torch.cat(video)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        video = video.cpu().float().numpy()
        return video

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        width,
        height,
        video_length,
        dtype,
        device,
        generator,
        latents=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            video_length,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def _encode_prompt(
        self,
        prompt,
        device,
        num_videos_per_prompt,
        do_classifier_free_guidance,
        negative_prompt,
    ):
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(
            prompt, padding="longest", return_tensors="pt"
        ).input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = self.tokenizer.batch_decode(
                untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
            )

        if (
            hasattr(self.text_encoder.config, "use_attention_mask")
            and self.text_encoder.config.use_attention_mask
        ):
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        text_embeddings = self.text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        text_embeddings = text_embeddings[0]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_videos_per_prompt, 1)
        text_embeddings = text_embeddings.view(
            bs_embed * num_videos_per_prompt, seq_len, -1
        )

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if (
                hasattr(self.text_encoder.config, "use_attention_mask")
                and self.text_encoder.config.use_attention_mask
            ):
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            uncond_embeddings = uncond_embeddings[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_videos_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(
                batch_size * num_videos_per_prompt, seq_len, -1
            )

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    @torch.no_grad()
    def __call__(
        self,
        ref_image,
        pose_images,
        width,
        height,
        video_length,
        num_inference_steps,
        guidance_scale=3.5,
        num_images_per_prompt=1,
        eta: float = 0.0,
        prompt = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        temporal_window_size = 4,
        temporal_adaptive_step = 4,
        latents = None,
        cond_proj_dim = 512,
        attn_masked = False,
        reference_kv_cache = False,
        overlap = False,
        vae_bsz = 1,
        vae_reference = False,
        pose_inject = None,
        **kwargs,
    ):
        assert num_inference_steps % temporal_adaptive_step == 0, "temporal_adaptive_step should be divisor of num_inference_steps"
        assert video_length % temporal_window_size == 0, "temporal_window_size should be divisor of video_length"
        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        device = self._execution_device

        do_classifier_free_guidance = False

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

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        jump = num_inference_steps // temporal_adaptive_step
        windows = video_length // temporal_window_size
        step_length = self.scheduler.config.num_train_timesteps // num_inference_steps

        batch_size = 1

        if attn_masked:
            attn_mask = rearrange(torch.tril(torch.ones((temporal_adaptive_step,temporal_adaptive_step))).unsqueeze(-1).repeat(1,1,temporal_window_size*temporal_window_size),"p q (u v) -> (p u) (q v)", u=temporal_window_size,v=temporal_window_size).transpose(0,1).unsqueeze(0).to(device)
        else:
            attn_mask = None

        # Prepare clip image embeds
        clip_image = self.clip_image_processor.preprocess(
            ref_image, return_tensors="pt"
        ).pixel_values
        clip_image_embeds = self.image_encoder(
            clip_image.to(device, dtype=self.image_encoder.dtype)
        ).image_embeds
        encoder_hidden_states = clip_image_embeds.unsqueeze(1)
        if(not(prompt is None)):
            embs = self._encode_prompt(prompt, encoder_hidden_states.device, 1, False, None)
            encoder_hidden_states = torch.cat([encoder_hidden_states, embs], dim=1)

        uncond_encoder_hidden_states = torch.zeros_like(encoder_hidden_states)

        if do_classifier_free_guidance:
            encoder_hidden_states = torch.cat(
                [uncond_encoder_hidden_states, encoder_hidden_states], dim=0
            )
        reference_control_writer = ReferenceAttentionControl(
            self.reference_unet,
            do_classifier_free_guidance=do_classifier_free_guidance,
            mode="write",
            batch_size=batch_size,
            fusion_blocks="full",
        )
        reference_control_reader = ReferenceAttentionControl(
            self.denoising_unet,
            do_classifier_free_guidance=do_classifier_free_guidance,
            mode="read",
            batch_size=batch_size,
            fusion_blocks="full",
            cache_kv=reference_kv_cache,
        )


        num_channels_latents = self.denoising_unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            width,
            height,
            video_length,
            clip_image_embeds.dtype,
            device,
            generator,
            latents = latents
        )
        w = guidance_scale
        if not isinstance(w, torch.Tensor):
            w = torch.tensor([w])
        w_embedding = guidance_scale_embedding(w, embedding_dim=cond_proj_dim)
        w_embedding = w_embedding.to(device=latents.device, dtype=latents.dtype)

        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # Prepare ref image latents
        ref_image_tensor = self.ref_image_processor.preprocess(
            ref_image, height=height, width=width
        )  # (bs, c, width, height)
        ref_image_tensor = ref_image_tensor.to(
            dtype=self.vae.dtype, device=self.vae.device
        )
        if(vae_reference):
            posterior, hiddens = self.vae.encode(ref_image_tensor, return_dict=False , return_hiddens=True)
            ref_image_latents = posterior.mean
        else:
            hiddens = None
            ref_image_latents = self.vae.encode(ref_image_tensor).latent_dist.mean

        ref_image_latents = ref_image_latents * 0.18215  # (b, 4, h, w)

        # Prepare a list of pose condition images
        pose_cond_tensor_list = []
        for pose_image in pose_images:
            pose_cond_tensor = (
                torch.from_numpy(np.array(pose_image.resize((width, height)))) / 255.0
            )
            pose_cond_tensor = pose_cond_tensor.permute(2, 0, 1).unsqueeze(
                1
            )  # (c, 1, h, w)
            pose_cond_tensor_list.append(pose_cond_tensor)
        pose_cond_tensor = torch.cat(pose_cond_tensor_list, dim=1)  # (c, t, h, w)
        pose_cond_tensor = pose_cond_tensor.unsqueeze(0)
        pose_cond_tensor = pose_cond_tensor.to(
            device=device, dtype=self.pose_guider.dtype
        )
        if pose_inject is None:
            pose_fea = self.pose_guider(pose_cond_tensor)
        else:
            pose_fea = self.pose_guider(pose_cond_tensor, pose_inject)
        pose_fea = (
            torch.cat([pose_fea] * 2) if do_classifier_free_guidance else pose_fea
        )

        # denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=windows + temporal_adaptive_step - 1) as progress_bar:
            self.reference_unet(
                    ref_image_latents.repeat(
                        (2 if do_classifier_free_guidance else 1), 1, 1, 1
                    ),
                    torch.zeros((batch_size,),dtype=torch.float32,device=ref_image_latents.device),
                    # t,
                    encoder_hidden_states=encoder_hidden_states,
                    return_dict=False,
                )
            reference_control_reader.update(reference_control_writer)
            for i in range(windows + temporal_adaptive_step - 1):
                l = max(0, i - temporal_adaptive_step + 1) * temporal_window_size
                r = min(windows, i + 1) * temporal_window_size
                for j in range(jump):
                    if overlap and r> temporal_adaptive_step * temporal_window_size:
                        latent_model_input = (
                            torch.cat([latents[:,:,l - temporal_window_size:r,:,:]] * 2) if do_classifier_free_guidance else latents[:,:,l - temporal_window_size:r,:,:]
                        )
                    else:
                        latent_model_input = (
                            torch.cat([latents[:,:,l:r,:,:]] * 2) if do_classifier_free_guidance else latents[:,:,l:r,:,:]
                        )
                    ut = self.scheduler.timesteps[
                        (torch.linspace(num_inference_steps,0, temporal_adaptive_step + 1)[1:].long())
                        [(-r // temporal_window_size if l==0 else 0):((windows - l // temporal_window_size) if r == video_length else temporal_adaptive_step)]
                    ] - step_length * j
                    if overlap and r> temporal_adaptive_step * temporal_window_size:
                        ut = torch.cat([torch.tensor([0], device=ut.device, dtype=ut.dtype), ut])
                    ut = torch.stack([torch.stack([ut] * temporal_window_size).T.reshape((-1))] * batch_size * (2 if do_classifier_free_guidance else 1)).to(device)
                    noise_pred = self.denoising_unet(
                        latent_model_input,
                        ut,
                        encoder_hidden_states=encoder_hidden_states,
                        pose_cond_fea=pose_fea[:,:,l-temporal_window_size:r,:,:] if overlap and r> temporal_adaptive_step * temporal_window_size else pose_fea[:,:,l:r,:,:],
                        timestep_cond=w_embedding,
                        return_dict=False,
                        attention_mask=attn_mask[:,:r-l,:r-l] if attn_masked else attn_mask,
                    )[0]
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (
                            noise_pred_text - noise_pred_uncond
                        )
                    if overlap and r> temporal_adaptive_step * temporal_window_size:
                        noise_pred = noise_pred[:,:,temporal_window_size:,:,:]
                        ut = ut[:,temporal_window_size:]
                    latents[:,:,l:r,:,:] = self.scheduler.step(
                        noise_pred, ut.chunk(2)[0], latents[:,:,l:r,:,:], **extra_step_kwargs, return_dict=False
                    )[0]
                progress_bar.update()

            reference_control_reader.clear()
            reference_control_writer.clear()

        # Post-processing
        images = self.decode_latents(latents, vae_bsz, hiddens=hiddens)  # (b, c, f, h, w)

        # Convert to tensor
        if output_type == "tensor":
            images = torch.from_numpy(images)

        if not return_dict:
            return images

        return Pose2VideoPipelineOutput(videos=images)
