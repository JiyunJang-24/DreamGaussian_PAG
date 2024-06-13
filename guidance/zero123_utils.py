from diffusers import DDIMScheduler
from typing import Optional
import torchvision.transforms.functional as TF

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.utils import deprecate
from diffusers.models.attention_processor import Attention, AttnProcessor2_0


import sys
sys.path.append('./')

from zero123 import Zero123Pipeline

class PAGCFGIdentitySelfAttnProcessor:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)
        
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        
        # chunk
        hidden_states_uncond, hidden_states_org, hidden_states_ptb = hidden_states.chunk(3)
        hidden_states_org = torch.cat([hidden_states_uncond, hidden_states_org])
        
        # original path
        batch_size, sequence_length, _ = hidden_states_org.shape

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states_org = attn.group_norm(hidden_states_org.transpose(1, 2)).transpose(1, 2)
        
        query = attn.to_q(hidden_states_org)
        key = attn.to_k(hidden_states_org)
        value = attn.to_v(hidden_states_org)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states_org = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states_org = hidden_states_org.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states_org = hidden_states_org.to(query.dtype)
        
        # linear proj
        hidden_states_org = attn.to_out[0](hidden_states_org)
        # dropout
        hidden_states_org = attn.to_out[1](hidden_states_org)

        if input_ndim == 4:
            hidden_states_org = hidden_states_org.transpose(-1, -2).reshape(batch_size, channel, height, width)

        # perturbed path (identity attention)
        batch_size, sequence_length, _ = hidden_states_ptb.shape

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states_ptb = attn.group_norm(hidden_states_ptb.transpose(1, 2)).transpose(1, 2)

        value = attn.to_v(hidden_states_ptb)
        hidden_states_ptb = value
        hidden_states_ptb = hidden_states_ptb.to(query.dtype)
        
        # linear proj
        hidden_states_ptb = attn.to_out[0](hidden_states_ptb)
        # dropout
        hidden_states_ptb = attn.to_out[1](hidden_states_ptb)

        if input_ndim == 4:
            hidden_states_ptb = hidden_states_ptb.transpose(-1, -2).reshape(batch_size, channel, height, width)

        # cat
        hidden_states = torch.cat([hidden_states_org, hidden_states_ptb])

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class Zero123(nn.Module):
    def __init__(self, device, fp16=True, t_range=[0.02, 0.98], model_key="ashawkey/zero123-xl-diffusers"):
        super().__init__()

        self.device = device
        self.fp16 = fp16
        self.dtype = torch.float16 if fp16 else torch.float32

        assert self.fp16, 'Only zero123 fp16 is supported for now.'

        # model_key = "ashawkey/zero123-xl-diffusers"
        # model_key = './model_cache/stable_zero123_diffusers'

        self.pipe = Zero123Pipeline.from_pretrained(
            model_key,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        ).to(self.device)

        # stable-zero123 has a different camera embedding
        self.use_stable_zero123 = 'stable' in model_key

        self.pipe.image_encoder.eval()
        self.pipe.vae.eval()
        self.pipe.unet.eval()
        self.pipe.clip_camera_projection.eval()

        self.vae = self.pipe.vae
        self.unet = self.pipe.unet

        self.pipe.set_progress_bar_config(disable=True)

        self.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps

        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience

        self.embeddings = None

    @torch.no_grad()
    def get_img_embeds(self, x):
        # x: image tensor in [0, 1]
        x = F.interpolate(x, (256, 256), mode='bilinear', align_corners=False)
        x_pil = [TF.to_pil_image(image) for image in x]
        x_clip = self.pipe.feature_extractor(images=x_pil, return_tensors="pt").pixel_values.to(device=self.device, dtype=self.dtype)
        c = self.pipe.image_encoder(x_clip).image_embeds
        v = self.encode_imgs(x.to(self.dtype)) / self.vae.config.scaling_factor
        self.embeddings = [c, v]
    
    def get_cam_embeddings(self, elevation, azimuth, radius, default_elevation=0):
        if self.use_stable_zero123:
            T = np.stack([np.deg2rad(elevation), np.sin(np.deg2rad(azimuth)), np.cos(np.deg2rad(azimuth)), np.deg2rad([90 + default_elevation] * len(elevation))], axis=-1)
        else:
            # original zero123 camera embedding
            T = np.stack([np.deg2rad(elevation), np.sin(np.deg2rad(azimuth)), np.cos(np.deg2rad(azimuth)), radius], axis=-1)
        T = torch.from_numpy(T).unsqueeze(1).to(dtype=self.dtype, device=self.device) # [8, 1, 4]
        return T

    @torch.no_grad()
    def refine(self, pred_rgb, elevation, azimuth, radius, 
               guidance_scale=5, steps=50, strength=0.8, default_elevation=0,
               do_pag=True, pag_scale=2.0, pag_applied_layers_index=['d4']
        ):

        batch_size = pred_rgb.shape[0]

        self.scheduler.set_timesteps(steps)

        if strength == 0:
            init_step = 0
            latents = torch.randn((1, 4, 32, 32), device=self.device, dtype=self.dtype)
        else:
            init_step = int(steps * strength)
            pred_rgb_256 = F.interpolate(pred_rgb, (256, 256), mode='bilinear', align_corners=False)
            latents = self.encode_imgs(pred_rgb_256.to(self.dtype))
            latents = self.scheduler.add_noise(latents, torch.randn_like(latents), self.scheduler.timesteps[init_step])

        T = self.get_cam_embeddings(elevation, azimuth, radius, default_elevation)
        cc_emb = torch.cat([self.embeddings[0].repeat(batch_size, 1, 1), T], dim=-1)
        cc_emb = self.pipe.clip_camera_projection(cc_emb)
        vae_emb = self.embeddings[1].repeat(batch_size, 1, 1, 1)
        
        if do_pag:
            cc_emb = torch.cat([cc_emb, torch.zeros_like(cc_emb), cc_emb], dim=0)
            vae_emb = torch.cat([vae_emb, torch.zeros_like(vae_emb), vae_emb], dim=0)
        else:
            cc_emb = torch.cat([cc_emb, torch.zeros_like(cc_emb)], dim=0)
            vae_emb = torch.cat([vae_emb, torch.zeros_like(vae_emb)], dim=0)
        
        replace_processor = PAGCFGIdentitySelfAttnProcessor()

        if do_pag:            
            down_layers = []
            mid_layers = []
            up_layers = []
            for name, module in self.unet.named_modules():
                if 'attn1' in name and 'to' not in name:
                    layer_type = name.split('.')[0].split('_')[0]
                    if layer_type == 'down':
                        down_layers.append(module)
                    elif layer_type == 'mid':
                        mid_layers.append(module)
                    elif layer_type == 'up':
                        up_layers.append(module)
                    else:
                        raise ValueError(f"Invalid layer type: {layer_type}")
                    
        for i, t in enumerate(self.scheduler.timesteps[init_step:]):
            
            if do_pag:
                x_in = torch.cat([latents] * 3)
            else:
                x_in = torch.cat([latents] * 2)
            
            t_in = t.view(1).to(self.device)
            
            if do_pag:
                replace_processor = PAGCFGIdentitySelfAttnProcessor()
                drop_layers = pag_applied_layers_index
                for drop_layer in drop_layers:
                    try:
                        if drop_layer[0] == 'd':
                            down_layers[int(drop_layer[1])].processor = replace_processor
                        elif drop_layer[0] == 'm':
                            mid_layers[int(drop_layer[1])].processor = replace_processor
                        elif drop_layer[0] == 'u':
                            up_layers[int(drop_layer[1])].processor = replace_processor
                        else:
                            raise ValueError(f"Invalid layer type: {drop_layer[0]}")
                    except IndexError:
                        raise ValueError(
                            f"Invalid layer index: {drop_layer}. Available layers: {len(down_layers)} down layers, {len(mid_layers)} mid layers, {len(up_layers)} up layers."
                        )

            noise_pred = self.unet(
                torch.cat([x_in, vae_emb], dim=1),
                t_in.to(self.unet.dtype),
                encoder_hidden_states=cc_emb,
            ).sample
            
            if do_pag:
                noise_pred_cond, noise_pred_uncond, noise_pred_perturb = noise_pred.chunk(3)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond) + pag_scale * (noise_pred_cond - noise_pred_perturb)
                
            else:    
                noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        if do_pag:
            drop_layers = pag_applied_layers_index
            for drop_layer in drop_layers:
                try:
                    if drop_layer[0] == 'd':
                        down_layers[int(drop_layer[1])].processor = AttnProcessor2_0()
                    elif drop_layer[0] == 'm':
                        mid_layers[int(drop_layer[1])].processor = AttnProcessor2_0()
                    elif drop_layer[0] == 'u':
                        up_layers[int(drop_layer[1])].processor = AttnProcessor2_0()
                    else:
                        raise ValueError(f"Invalid layer type: {drop_layer[0]}")
                except IndexError:
                    raise ValueError(
                        f"Invalid layer index: {drop_layer}. Available layers: {len(down_layers)} down layers, {len(mid_layers)} mid layers, {len(up_layers)} up layers."
                    )
                    
                    
        imgs = self.decode_latents(latents) # [1, 3, 256, 256]
        return imgs
    
    def train_step(self, pred_rgb, elevation, azimuth, radius, step_ratio=None, guidance_scale=5, as_latent=False, default_elevation=0, 
                   do_pag=False, pag_applied_layers_index=['d4'], pag_scale=2.0):
        # pred_rgb: tensor [1, 3, H, W] in [0, 1]

        batch_size = pred_rgb.shape[0]

        if as_latent:
            latents = F.interpolate(pred_rgb, (32, 32), mode='bilinear', align_corners=False) * 2 - 1
        else:
            pred_rgb_256 = F.interpolate(pred_rgb, (256, 256), mode='bilinear', align_corners=False)
            latents = self.encode_imgs(pred_rgb_256.to(self.dtype))

        if step_ratio is not None:
            # dreamtime-like
            # t = self.max_step - (self.max_step - self.min_step) * np.sqrt(step_ratio)
            t = np.round((1 - step_ratio) * self.num_train_timesteps).clip(self.min_step, self.max_step)
            t = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
        else:
            t = torch.randint(self.min_step, self.max_step + 1, (batch_size,), dtype=torch.long, device=self.device)

        w = (1 - self.alphas[t]).view(batch_size, 1, 1, 1)

        with torch.no_grad():
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            
            if do_pag:
                x_in = torch.cat([latents_noisy] * 3)
                t_in = torch.cat([t] * 3)
            else:
                x_in = torch.cat([latents_noisy] * 2)
                t_in = torch.cat([t] * 2)


            T = self.get_cam_embeddings(elevation, azimuth, radius, default_elevation)
            cc_emb = torch.cat([self.embeddings[0].repeat(batch_size, 1, 1), T], dim=-1)
            cc_emb = self.pipe.clip_camera_projection(cc_emb)
            vae_emb = self.embeddings[1].repeat(batch_size, 1, 1, 1)
            
            if do_pag:
                cc_emb = torch.cat([cc_emb, torch.zeros_like(cc_emb), cc_emb], dim=0)
                vae_emb = torch.cat([vae_emb, torch.zeros_like(vae_emb), vae_emb], dim=0)
            else:
                cc_emb = torch.cat([cc_emb, torch.zeros_like(cc_emb)], dim=0)
                vae_emb = torch.cat([vae_emb, torch.zeros_like(vae_emb)], dim=0)
                
            if do_pag:            
                down_layers = []
                mid_layers = []
                up_layers = []
                for name, module in self.unet.named_modules():
                    if 'attn1' in name and 'to' not in name:
                        layer_type = name.split('.')[0].split('_')[0]
                        if layer_type == 'down':
                            down_layers.append(module)
                        elif layer_type == 'mid':
                            mid_layers.append(module)
                        elif layer_type == 'up':
                            up_layers.append(module)
                        else:
                            raise ValueError(f"Invalid layer type: {layer_type}")
            
                replace_processor = PAGCFGIdentitySelfAttnProcessor()
                drop_layers = pag_applied_layers_index
                for drop_layer in drop_layers:
                    try:
                        if drop_layer[0] == 'd':
                            down_layers[int(drop_layer[1])].processor = replace_processor
                        elif drop_layer[0] == 'm':
                            mid_layers[int(drop_layer[1])].processor = replace_processor
                        elif drop_layer[0] == 'u':
                            up_layers[int(drop_layer[1])].processor = replace_processor
                        else:
                            raise ValueError(f"Invalid layer type: {drop_layer[0]}")
                    except IndexError:
                        raise ValueError(
                            f"Invalid layer index: {drop_layer}. Available layers: {len(down_layers)} down layers, {len(mid_layers)} mid layers, {len(up_layers)} up layers."
                        )         
            
            noise_pred = self.unet(
                torch.cat([x_in, vae_emb], dim=1),
                t_in.to(self.unet.dtype),
                encoder_hidden_states=cc_emb,
            ).sample

        # noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
        # noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        
            if do_pag:
                noise_pred_cond, noise_pred_uncond, noise_pred_perturb = noise_pred.chunk(3)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond) + pag_scale * (noise_pred_cond - noise_pred_perturb)
                
            else:    
                noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                
            if do_pag:
                drop_layers = pag_applied_layers_index
                for drop_layer in drop_layers:
                    try:
                        if drop_layer[0] == 'd':
                            down_layers[int(drop_layer[1])].processor = AttnProcessor2_0()
                        elif drop_layer[0] == 'm':
                            mid_layers[int(drop_layer[1])].processor = AttnProcessor2_0()
                        elif drop_layer[0] == 'u':
                            up_layers[int(drop_layer[1])].processor = AttnProcessor2_0()
                        else:
                            raise ValueError(f"Invalid layer type: {drop_layer[0]}")
                    except IndexError:
                        raise ValueError(
                            f"Invalid layer index: {drop_layer}. Available layers: {len(down_layers)} down layers, {len(mid_layers)} mid layers, {len(up_layers)} up layers."
                        )

        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        target = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents.float(), target, reduction='sum')

        return loss
    

    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents

        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def encode_imgs(self, imgs, mode=False):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        if mode:
            latents = posterior.mode()
        else:
            latents = posterior.sample() 
        latents = latents * self.vae.config.scaling_factor

        return latents
    
    
if __name__ == '__main__':
    import cv2
    import argparse
    import numpy as np
    import matplotlib.pyplot as plt
    import kiui

    parser = argparse.ArgumentParser()

    parser.add_argument('input', type=str)
    parser.add_argument('--elevation', type=float, default=0, help='delta elevation angle in [-90, 90]')
    parser.add_argument('--azimuth', type=float, default=0, help='delta azimuth angle in [-180, 180]')
    parser.add_argument('--radius', type=float, default=0, help='delta camera radius multiplier in [-0.5, 0.5]')
    parser.add_argument('--stable', action='store_true')

    opt = parser.parse_args()

    device = torch.device('cuda')

    print(f'[INFO] loading image from {opt.input} ...')
    image = kiui.read_image(opt.input, mode='tensor')
    image = image.permute(2, 0, 1).unsqueeze(0).contiguous().to(device)
    image = F.interpolate(image, (256, 256), mode='bilinear', align_corners=False)

    print(f'[INFO] loading model ...')
    
    if opt.stable:
        zero123 = Zero123(device, model_key='ashawkey/stable-zero123-diffusers')
    else:
        zero123 = Zero123(device, model_key='ashawkey/zero123-xl-diffusers')

    print(f'[INFO] running model ...')
    zero123.get_img_embeds(image)

    azimuth = opt.azimuth
    while True:
        outputs = zero123.refine(image, elevation=[opt.elevation], azimuth=[azimuth], radius=[opt.radius], strength=0)
        plt.imshow(outputs.float().cpu().numpy().transpose(0, 2, 3, 1)[0])
        plt.show()
        azimuth = (azimuth + 10) % 360