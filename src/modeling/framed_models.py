import torch
from torch import nn
from src.taesdv.taesdv import MemBlock

class unet_work(nn.Module): # Ugly Power Strip
    def __init__(self, pose_guider, unet, vae, scheduler, timestep, w_embedding):
        super().__init__()
        self.pose_guider = pose_guider
        self.unet = unet
        self.vae = vae
        self.scheduler = scheduler
        self.scheduler.set_timesteps(4)
        self.timestep = timestep
        self.w_embedding = w_embedding

    def decode_slice(self, vae, x, pmems):
        assert x.ndim == 5, f"TAESDV operates on NTCHW tensors, but got {x.ndim}-dim tensor"
        N, T, C, H, W = x.shape
        assert C == 4, f"TAESDV decodes 4-channel latent tensors, but got {C}-channel tensor"
        x = x.reshape(N * T, C, H, W)
        omems = []
        i = 0
        for b in vae.decoder:
            if isinstance(b, MemBlock):
                _NT, C, H, W = x.shape
                pmem = pmems[i]
                i = i + 1
                mem = torch.cat([pmem[:,None,:,:,:], x.reshape(N, T, C, H, W)], dim=1)
                omems.append(mem[:,-1,:,:,:])
                mem = mem[:,:T].reshape(x.shape)
                x = b(x, mem)
            else:
                x = b(x)
        _NT, C, H, W = x.shape
        return x.view(N, T, C, H, W), omems
        
    def forward(self, sample, encoder_hidden_states, pose_cond_fea, pose, add_noise, new_noise, 
        d00, d01, d10, d11, d20, d21, m, u10, u11, u12, u20, u21, u22, u30, u31, u32, 
        pmem0, pmem1, pmem2, pmem3, pmem4, pmem5, pmem6, pmem7, pmem8, pmem9,
        ):
        new_pose_cond_fea = self.pose_guider(pose)
        #pose_cond_fea = new_pose_cond_fea.repeat(1,1,4,1,1)
        pose_cond_fea = torch.cat([pose_cond_fea, new_pose_cond_fea], dim=2)
        score = self.unet(sample, self.timestep, encoder_hidden_states, pose_cond_fea, self.w_embedding, d00, d01, d10, d11, d20, d21, m, u10, u11, u12, u20, u21, u22, u30, u31, u32)
        result = self.scheduler.step(
            score, self.timestep, sample, noise=add_noise, return_dict=False
        )[0].to(torch.float16)
        l_, mems = self.decode_slice(self.vae, result[:, :, :4, :, :].permute(0, 2, 1, 3, 4),[pmem0, pmem1, pmem2, pmem3, pmem4, pmem5, pmem6, pmem7, pmem8, pmem9])
        return l_.clamp(0, 1), torch.cat([result[:, :, 4:, :, :], new_noise], dim=2), pose_cond_fea[:, :, 4:, :, :], mems[0], mems[1], mems[2], mems[3], mems[4], mems[5], mems[6], mems[7], mems[8], mems[9]
    
    def get_sample_input(self, batchsize, height, width, dtype, device):
        tw, ts, tb = 4, 4, 16 # temporal window size| temporal adaptive steps | temporal batch size
        b, h, w = batchsize, height, width
        lh, lw = height // 8, width // 8 # latent height | width
        cd0, cd1, cd2, cm, cu1, cu2, cu3 = 320, 640, 1280, 1280, 1280, 640, 320 # unet channels
        emb, tvc = 768, 64 # CLIP Embedding Dims | TAESDV Channels
        lc, ic = 4, 3 # latent | image channels
        profile = {
            "sample" : [b, lc, tb, lh, lw],
            "encoder_hidden_states" : [b, 1, emb],
            "pose_cond_fea" : [b, cd0, tw * (ts - 1), lh, lw],
            "pose" : [b, ic, tw, h, w],
            "add_noise" : [b, lc, tb, lh, lw],
            "new_noise" : [b, lc, tw, lh, lw],
            "d00" : [b, lh * lw, cd0],
            "d01" : [b, lh * lw, cd0],
            "d10" : [b, lh * lw // 4, cd1],
            "d11" : [b, lh * lw // 4, cd1],
            "d20" : [b, lh * lw // 16, cd2],
            "d21" : [b, lh * lw // 16, cd2],
            "m" : [b, lh * lw // 64, cm],
            "u10" : [b, lh * lw // 16, cu1],
            "u11" : [b, lh * lw // 16, cu1],
            "u12" : [b, lh * lw // 16, cu1],
            "u20" : [b, lh * lw // 4, cu2],
            "u21" : [b, lh * lw // 4, cu2],
            "u22" : [b, lh * lw // 4, cu2],
            "u30" : [b, lh * lw, cu3],
            "u31" : [b, lh * lw, cu3],
            "u32" : [b, lh * lw, cu3],
            "pmem0" : [b, tvc, lh, lw],
            "pmem1" : [b, tvc, lh, lw],
            "pmem2" : [b, tvc, lh, lw],
            "pmem3" : [b, tvc, lh * 2, lw * 2],
            "pmem4" : [b, tvc, lh * 2, lw * 2],
            "pmem5" : [b, tvc, lh * 2, lw * 2],
            "pmem6" : [b, tvc, lh * 4, lw * 4],
            "pmem7" : [b, tvc, lh * 4, lw * 4],
            "pmem8" : [b, tvc, lh * 4, lw * 4],
            "pmem9" : [b, tvc, h, w],
        }
        return {k: torch.randn(profile[k], dtype=dtype, device=device) for k in profile}
    
    def get_input_names(self):
        return ["sample", "encoder_hidden_states", "pose_cond_fea", "pose", "add_noise", "new_noise", "d00", "d01", "d10", "d11", "d20", "d21", "m", "u10", "u11", "u12", "u20", "u21", "u22", "u30", "u31", "u32", "pmem0", "pmem1", "pmem2", "pmem3", "pmem4", "pmem5", "pmem6", "pmem7", "pmem8", "pmem9"]
   
    def get_output_names(self):
        return ["image", "latents", "pose_cond_fea_out", "omem0", "omem1", "omem2", "omem3", "omem4", "omem5", "omem6", "omem7", "omem8", "omem9"]

    def get_dynamic_axes(self):
        return None
    
class reference_net(nn.Module):
    def __init__(self, reference_unet, reference_control_writer):
        super().__init__()
        
        self.reference_unet = reference_unet
        self.reference_control_writer = reference_control_writer
        
    def forward(self, latents, clip_embeds):
        self.reference_unet(latents,torch.zeros((1),dtype=torch.float16,device="cuda"),encoder_hidden_states=clip_embeds)
        z = self.reference_control_writer.output()
        return z["d00"], z["d01"], z["d10"], z["d11"], z["d20"], z["d21"], z["m"], z["u10"], z["u11"], z["u12"], z["u20"], z["u21"], z["u22"], z["u30"], z["u31"], z["u32"]
    
    def get_sample_input(self, batchsize, height, width, dtype, device):
        tw, ts, tb = 4, 4, 16 # temporal window size| temporal adaptive steps | temporal batch size
        b, h, w = batchsize, height, width
        lh, lw = height // 8, width // 8 # latent height | width
        cd0, cd1, cd2, cm, cu1, cu2, cu3 = 320, 640, 1280, 1280, 1280, 640, 320 # unet channels
        emb, tvc = 768, 64 # CLIP Embedding Dims | TAESDV Channels
        lc, ic = 4, 3 # latent | image channels
        profile = {
            "latents" : [batchsize, lc, lh, lw],
            "clip_embeds" : [batchsize, 1, emb],
        }
        return {k: torch.randn(profile[k], dtype=dtype, device=device) for k in profile}

    def get_input_names(self):
        return ["latents", "clip_embeds"]
    
    def get_output_names(self):
        return ["d00", "d01", "d10", "d11", "d20", "d21", "m", "u10", "u11", "u12", "u20", "u21", "u22", "u30", "u31", "u32"]

    def get_dynamic_axes(self):
        return None

class vae_encoder(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, images):
        return self.vae.encode(images).latent_dist.mean * 0.18215 
    
    def get_sample_input(self, batchsize, height, width, dtype, device):
        tw, ts, tb = 4, 4, 16 # temporal window size| temporal adaptive steps | temporal batch size
        b, h, w = batchsize, height, width
        lh, lw = height // 8, width // 8 # latent height | width
        cd0, cd1, cd2, cm, cu1, cu2, cu3 = 320, 640, 1280, 1280, 1280, 640, 320 # unet channels
        emb, tvc = 768, 64 # CLIP Embedding Dims | TAESDV Channels
        lc, ic = 4, 3 # latent | image channels
        profile = {
            "images" : [batchsize, ic, h, w],
        }
        return {k: torch.randn(profile[k], dtype=dtype, device=device) for k in profile}
    
    def get_input_names(self):
        return ["images",]
    
    def get_output_names(self):
        return ["latents"]

    def get_dynamic_axes(self):
        return None
    
class clip_model(nn.Module):
    def __init__(self, image_encoder):
        super().__init__()
        self.image_encoder = image_encoder

    def forward(self, images):
        return self.image_encoder(images).image_embeds
    
    def get_sample_input(self, batchsize, height, width, dtype, device):
        tw, ts, tb = 4, 4, 16 # temporal window size| temporal adaptive steps | temporal batch size
        b, h, w = batchsize, height, width
        lh, lw = height // 8, width // 8 # latent height | width
        cd0, cd1, cd2, cm, cu1, cu2, cu3 = 320, 640, 1280, 1280, 1280, 640, 320 # unet channels
        emb, tvc = 768, 64 # CLIP Embedding Dims | TAESDV Channels
        lc, ic = 4, 3 # latent | image channels
        profile = {
            "images" : [batchsize, ic, 224, 224],
        }
        return {k: torch.randn(profile[k], dtype=dtype, device=device) for k in profile}
    
    def get_input_names(self):
        return ["images",]
    
    def get_output_names(self):
        return ["clip_embeds"]

    def get_dynamic_axes(self):
        return None