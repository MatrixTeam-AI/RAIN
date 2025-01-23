from src.pipeline.pipeline_pose2vid_lcm import Pose2VideoPipelineLCM
from omegaconf import OmegaConf
import torch
from src.modeling.translation import translation
from src.dwpose import DWposeDetector
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.pose_guider import PoseGuider
from src.models.unet_3d import UNet3DConditionModel
from src.scheduler.scheduler_lcm import LCMScheduler
from diffusers import AutoencoderKL
from transformers import CLIPVisionModelWithProjection
from PIL import Image
import numpy as np
import cv2 
from decord import VideoReader


config_path = "./configs/rain_morpher.yaml"
cfg = OmegaConf.load(config_path)
device = torch.device("cuda")
dtype = torch.float16
original_video_path = "./test_videos/test1.mp4" # change to your video path
reference_image_path = "./example_images/test1.png"
output_video_path = "./output.mp4"

###### Load Models ######
vae = AutoencoderKL.from_pretrained(cfg.vae_model_path).to(device=device, dtype=dtype)
image_encoder = CLIPVisionModelWithProjection.from_pretrained(cfg.image_encoder_path,).to(dtype=dtype, device=device)

reference_unet = UNet2DConditionModel.from_config(
    cfg.unet_config_path,
).to("cuda",dtype=torch.float16)
reference_unet_state_dict = torch.load(cfg.reference_unet_weight_path, map_location="cpu")
reference_unet.load_state_dict(reference_unet_state_dict)
del reference_unet_state_dict

pose_guider = PoseGuider(
    conditioning_embedding_channels=320, block_out_channels=(16, 32, 96, 256)
).to(device=device, dtype=dtype)
pose_guider_state_dict = torch.load(cfg.pose_guider_weight_path, map_location="cpu")
pose_guider.load_state_dict(pose_guider_state_dict)
del pose_guider_state_dict

denoising_unet = UNet3DConditionModel.from_pretrained_2d(
    cfg.denoising_unet_weight_path,
    unet_additional_kwargs=OmegaConf.to_container(
        cfg.unet_additional_kwargs
    ),
    direct_weight=True,
    config_path=cfg.unet_config_path
).to(device=device, dtype=dtype)

scheduler = LCMScheduler(**cfg.noise_scheduler_kwargs)
scheduler.to(device)


pipe = Pose2VideoPipelineLCM(vae,image_encoder,reference_unet,denoising_unet,pose_guider,scheduler).to(device) # setup pipeline

###### config facial landmark detector

params = dict(face_length_ratio=0.75,
eye_height_ratio=3.0,
eye_width_ratio=1.6,
mouth_height_ratio=1.2,
mouth_width_ratio=0.67,
jaw_width_ratio=0.9,
eye_horizontal_correction=0.15,
eye_distance_correction=0.05,
eye_vertical_correction=0.10,
mouth_horizontal_correction=0.15,
mouth_vertical_correction=0.03,
nose_horizontal_correction=0.1,
nose_vertical_correction=0.05,
)

def param2translations(face_length_ratio=0.75, eye_height_ratio=3.0, eye_width_ratio=1.6, mouth_height_ratio=1.2, mouth_width_ratio=0.67, jaw_width_ratio=0.9, eye_horizontal_correction=0.15, eye_distance_correction=0.05, eye_vertical_correction=0.10, mouth_horizontal_correction=0.15, mouth_vertical_correction=0.0, nose_vertical_correction=0.0,nose_horizontal_correction=0.0):
    translations = ([
        translation(0, 68, vertical_scale=face_length_ratio, xaxis=(2, 14)),
        translation(36, 42,vertical_scale=eye_height_ratio, xaxis=(0,3), horizontal_scale=eye_width_ratio,),
        translation(42, 48,vertical_scale=eye_height_ratio, xaxis=(0,3), horizontal_scale=eye_width_ratio),
        translation(60, 68,vertical_scale=mouth_height_ratio, xaxis=(0, 4), horizontal_scale=mouth_width_ratio),
        translation(36, 42, xaxis=(0,3), relative_length=(33, 13), horizontal_transition=eye_horizontal_correction - eye_distance_correction),
        translation(42, 48, xaxis=(0,3), relative_length=(33, 13), horizontal_transition=eye_horizontal_correction),
        translation(36, 42, xaxis=(0,3), relative_length=(33, 3), horizontal_transition=-eye_horizontal_correction),
        translation(42, 48, xaxis=(0,3), relative_length=(33, 3), horizontal_transition=-(eye_horizontal_correction - eye_distance_correction)),
        translation(36, 42, xaxis=(0,3), relative_length=(19, 6), vertical_transition=eye_vertical_correction),
        translation(42, 48, xaxis=(0,3), relative_length=(24, 10), vertical_transition=eye_vertical_correction),
        translation(60, 68, xaxis=(0,4), relative_length=(33, 13), horizontal_transition=mouth_horizontal_correction),
        translation(60, 68, xaxis=(0,4), relative_length=(33, 3), horizontal_transition=-mouth_horizontal_correction),
        translation(60, 68, relative_length=(27, 8), xaxis=(0, 4), vertical_transition=mouth_vertical_correction),
        translation(27, 36, relative_length=(27, 8), xaxis=(4, 8), vertical_transition=nose_vertical_correction),
        translation(27, 36, relative_length=(22, 16), xaxis=(4, 8), horizontal_transition=nose_horizontal_correction),
        translation(27, 36, relative_length=(21, 0), xaxis=(4, 8), horizontal_transition=-nose_horizontal_correction),
        translation(4, 13, xaxis=(0,8), horizontal_scale=jaw_width_ratio),
    ])
    return translations

translations = param2translations(**params)
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
face_tracker = DWposeDetector(cfg.dwpose_detection_model_path, cfg.dwpose_landmark_model_path, ( [('CUDAExecutionProvider', {'device_id': 0,})]), face_only=True, translations=translations, detection_size=(512, 512))
face_tracker.set_patterns(patterns)
face_tracker.set_translations(translations)



###### get pose sequence from video

vr = VideoReader(original_video_path) # VideoReader from decord

ps = [Image.fromarray(face_tracker.detect(f.asnumpy())) for f in vr]



def save_video( imgs, vid_file_name, fps=24):
    video_codec = cv2.VideoWriter_fourcc(*"mp4v")
    segment_file = cv2.VideoWriter(vid_file_name, video_codec, fps, (imgs[0].shape[1], imgs[0].shape[0]))
    if(imgs[0].dtype == np.uint8):
        for f in imgs:
            segment_file.write(f[:,:,::-1])
    else:
        for f in imgs:
            segment_file.write(np.uint8(255*f)[:,:,::-1])
    
    segment_file.release()

#### generation

video = pipe(
    ref_image=Image.open(reference_image_path),
    pose_images=ps,
    width=512,
    height=512,
    num_inference_steps=4,
    guidance_scale=3.5,
    video_length=100
)

save_video(video[0].permute(0,2,3,4,1)[0].cpu().numpy(), output_video_path)