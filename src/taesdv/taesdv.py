#!/usr/bin/env python3
"""
Tiny AutoEncoder for Stable Diffusion Videos
(DNN for encoding / decoding videos to SD's latent space)
"""
import torch
import torch.nn as nn
from collections import namedtuple

DecoderResult = namedtuple("DecoderResult", ("frame", "memory"))

def conv(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)

class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3

class Block(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(conv(n_in, n_out), nn.ReLU(), conv(n_out, n_out), nn.ReLU(), conv(n_out, n_out))
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.fuse = nn.ReLU()
    def forward(self, x):
        return self.fuse(self.conv(x) + self.skip(x))

class MemBlock(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(conv(n_in * 2, n_out), nn.ReLU(), conv(n_out, n_out), nn.ReLU(), conv(n_out, n_out))
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.act = nn.ReLU()
    def forward(self, x, mem):
        return self.act(self.conv(torch.cat([x, mem], 1)) + self.skip(x))

class TAESDV(nn.Module):
    def __init__(self, checkpoint_path="taesdv.pth"):
        """Initialize pretrained TAESDV on the given device from the given checkpoints."""
        super().__init__()
        self.encoder = nn.Sequential(
            conv(3, 64), Block(64, 64),
            conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
            conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
            conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
            conv(64, 4),
        )
        self.decoder = nn.Sequential(
            Clamp(), conv(4, 64), nn.ReLU(),
            MemBlock(64, 64), MemBlock(64, 64), MemBlock(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
            MemBlock(64, 64), MemBlock(64, 64), MemBlock(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
            MemBlock(64, 64), MemBlock(64, 64), MemBlock(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
            MemBlock(64, 64), conv(64, 3),
        )
        if checkpoint_path is not None:
            self.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))

    def encode_frame(self, x):
        """Encode a single RGB timestep to latents.

        Args:
            x: input NCHW RGB (C=3) tensor with values in [0, 1].
        Returns NCHW latent tensor with ~Gaussian values.
        """
        assert x.ndim == 4 and x.shape[1] == 3, f"Could not encode frame of shape {x.shape}"
        return self.encoder(x)

    def decode_frame(self, x, mem=None):
        """Decode a single latent timestep to RGB.

        Args:
            x: input NCHW latent (C=4) tensor with ~Gaussian values.
            mem: recurrent memory tensor. Should be:
                None if this is the first decoded frame, or
                memory from previous step if this a subsequent decoded frame.

        Returns a dictionary of:
            frame: NCHW RGB (C=3) decoded video frame with ~[0, 1] values
            memory: memory for decoding subsequent frames.
        """
        assert x.ndim == 4 and x.shape[1] == 4, f"Could not decode frame of shape {x.shape}"
        out_mem, in_mem = [], None if mem is None else list(mem)
        for b in self.decoder:
            if isinstance(b, MemBlock):
                out_mem.append(x)
                x = b(x, x * 0 if in_mem is None else in_mem.pop(0))
            else:
                x = b(x)
        return DecoderResult(x, out_mem)

    def encode_video(self, x, parallel=True):
        """Encode a sequence of frames.

        Args:
            x: input NTCHW RGB (C=3) tensor with values in [0, 1].
            parallel: if True, all frames will be processed at once.
              (this is faster but may require more memory).
              if False, frames will be processed sequentially.
        Returns NTCHW latent tensor with ~Gaussian values.
        """
        assert x.ndim == 5, f"TAESDV operates on NTCHW tensors, but got {x.ndim}-dim tensor"
        N, T, C, H, W = x.shape
        assert C == 3, f"TAESDV encodes RGB tensors, but got {C}-channel tensor"
        if parallel:
            x = self.encode_frame(x.reshape(N*T, C, H, W))
            return x.view(N, T, *x.shape[1:])
        else:
            return torch.stack([self.encode_frame(frame) for frame in x.view(N, T*C, H, W).chunk(T, dim=1)], 1)

    def decode_video(self, x, parallel=True):
        """Decode a sequence of frames.

        Args:
            x: input NTCHW latent (C=4) tensor with ~Gaussian values.
            parallel: if true, all frames will be processed at once.
              (this is faster but may require more memory).
        Returns NTCHW RGB tensor with ~[0, 1] values.
        """
        assert x.ndim == 5, f"TAESDV operates on NTCHW tensors, but got {x.ndim}-dim tensor"
        N, T, C, H, W = x.shape
        assert C == 4, f"TAESDV decodes 4-channel latent tensors, but got {C}-channel tensor"
        if parallel:
            x = x.reshape(N*T, C, H, W)
            for b in self.decoder:
                if isinstance(b, MemBlock):
                    _NT, C, H, W = x.shape
                    # mem is just the current input shifted 1 frame forward along time axis
                    mem = torch.nn.functional.pad(x.reshape(N, T, C, H, W), (0,0,0,0,0,0,1,0), value=0)[:,:T].reshape(x.shape)
                    x = b(x, mem)
                else:
                    x = b(x)
            _NT, C, H, W = x.shape
            return x.view(N, T, C, H, W)
        else:
            # if you're running TAESDV in an interactive / real-time loop,
            # this is how you run it.
            out, mem = [], None
            for latent in x.reshape(N, T * C, H, W).chunk(T, dim=1):
                frame, mem = self.decode_frame(latent, mem)
                out.append(frame)
            return torch.stack(out, 1)

    def decode_slice(self, x, pmems):
        assert x.ndim == 5, f"TAESDV operates on NTCHW tensors, but got {x.ndim}-dim tensor"
        N, T, C, H, W = x.shape
        assert C == 4, f"TAESDV decodes 4-channel latent tensors, but got {C}-channel tensor"
        x = x.reshape(N*T, C, H, W)
        omems = []
        i = 0
        for b in self.decoder:
            if isinstance(b, MemBlock):
                _NT, C, H, W = x.shape
                pmem = pmems[i]
                i = i + 1
                # mem is just the current input shifted 1 frame forward along time axis
                mem = torch.cat([pmem[:,None,:,:,:], x.reshape(N, T, C, H, W)], dim=1)
                omems.append(mem[:,-1,:,:,:])
                mem = mem[:,:T].reshape(x.shape)
                x = b(x, mem)
            else:
                x = b(x)
        _NT, C, H, W = x.shape
        return x.view(N, T, C, H, W), omems


@torch.no_grad()
def main():
    """Run TAESDV roundtrip reconstruction on the given video paths."""
    import sys
    import cv2 # no highly esteemed deed is commemorated here

    class VideoTensorReader:
        def __init__(self, video_file_path):
            self.cap = cv2.VideoCapture(video_file_path)
            assert self.cap.isOpened(), f"Could not load {video_file_path}"
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        def __iter__(self):
            return self
        def __next__(self):
            ret, frame = self.cap.read()
            if not ret:
                self.cap.release()
                raise StopIteration  # End of video or error
            return torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).permute(2, 0, 1) # BGR HWC -> RGB CHW

    class VideoTensorWriter:
        def __init__(self, video_file_path, width_height, fps=30):
            self.writer = cv2.VideoWriter(video_file_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, width_height)
            assert self.writer.isOpened(), f"Could not create writer for {video_file_path}"
        def write(self, frame_tensor):
            assert frame_tensor.ndim == 3 and frame_tensor.shape[0] == 3, f"{frame_tensor.shape}??"
            self.writer.write(cv2.cvtColor(frame_tensor.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR)) # RGB CHW -> BGR HWC
        def __del__(self):
            if hasattr(self, 'writer'): self.writer.release()

    dev = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float16
    print("Using device", dev, "and dtype", dtype)
    taesdv = TAESDV().to(dev, dtype)
    for video_path in sys.argv[1:]:
        print(f"Processing {video_path}...")
        # TAESDV supports both parallel (fast, memory-heavy) and serial (slower, memory-light) decoding strategies.
        # we'll pick one based on the video length.
        video_in = VideoTensorReader(video_path)
        video = torch.stack(list(video_in), 0)[None]
        vid_dev = video.to(dev, dtype).div_(255.0)
        if video.numel() < 100_000_000:
            print(f"  {video_path} seems small enough, will process all frames in parallel")
            # convert to device tensor
            vid_enc = taesdv.encode_video(vid_dev)
            print(f"  Encoded {video_path}")
            vid_dec = taesdv.decode_video(vid_enc)
            print(f"  Decoded {video_path}")
        else:
            print(f"  {video_path} seems large, will process each frame sequentially")
            # convert to device tensor
            vid_enc = taesdv.encode_video(vid_dev, parallel=False)
            print(f"  Encoded {video_path}")
            vid_dec = taesdv.decode_video(vid_enc, parallel=False)
            print(f"  Decoded {video_path}")
        video_out = VideoTensorWriter(video_path + ".reconstructed.mp4", (vid_dec.shape[-1], vid_dec.shape[-2]), fps=int(round(video_in.fps)))
        for frame in vid_dec.clamp_(0, 1).mul_(255).round_().byte().cpu()[0]:
            video_out.write(frame)

if __name__ == "__main__":
    main()
