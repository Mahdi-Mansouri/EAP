from omegaconf import OmegaConf
import torch
from PIL import Image
from torchvision import transforms
import os
from tqdm import tqdm
from einops import rearrange
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import einops

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config
import random
import glob
import re
import shutil
import pdb
import argparse
from convertModels import savemodelDiffusers
from PIL import Image
from torch.autograd import Variable
from utils_exp import get_prompt
from utils_alg import load_img, moving_average, plot_loss, get_models, save_to_dict
from gen_embedding_matrix import learn_k_means_from_input_embedding, learn_k_means_from_output, save_embedding_matrix, search_closest_tokens, retrieve_embedding_token


# Util Functions
def load_model_from_config(config, ckpt, device="cpu", verbose=False):
    """Loads a model from config and a ckpt"""
    if isinstance(config, (str, Path)):
        config = OmegaConf.load(config)

    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    model.cond_stage_model.device = device
    return model


@torch.no_grad()
def sample_model(model, sampler, c, h, w, ddim_steps, scale, ddim_eta, start_code=None, n_samples=1,t_start=-1,log_every_t=None,till_T=None,verbose=True):
    """Sample the model"""
    uc = None
    if scale != 1.0:
        uc = model.get_learned_conditioning(n_samples * [""])
    log_t = 100
    if log_every_t is not None:
        log_t = log_every_t
    shape = [4, h // 8, w // 8]
    samples_ddim, inters = sampler.sample(S=ddim_steps,
                                     conditioning=c,
                                     batch_size=n_samples,
                                     shape=shape,
                                     verbose=False,
                                     x_T=start_code,
                                     unconditional_guidance_scale=scale,
                                     unconditional_conditioning=uc,
                                     eta=ddim_eta,
                                     verbose_iter = verbose,
                                     t_start=t_start,
                                     log_every_t = log_t,
                                     till_T = till_T
                                    )
    if log_every_t is not None:
        return samples_ddim, inters
    return samples_ddim

def train(prompt, train_method, start_guidance, negative_guidance, iterations, lr, config_path, ckpt_path, diffusers_config_path, devices, seperator=None, image_size=512, ddim_steps=50, args=None):
    # PROMPT CLEANING
    word_print = prompt.replace(' ','')

    prompt, preserved = get_prompt(prompt)

    if seperator is not None:
        erased_words = prompt.split(seperator)
        erased_words = [word.strip() for word in erased_words]
        preserved_words = preserved.split(seperator)
        preserved_words = [word.strip() for word in preserved_words]
    else:
        erased_words = [prompt]
        preserved_words = [preserved]
    
    print('to be erased:', erased_words)
    print('to be preserved:', preserved_words)
    preserved_words.append('')

    ddim_eta = 0
    
    # --- LOAD MODELS ---
    print(f"Loading Student Model to {devices[0]}")
    model = load_model_from_config(config_path, ckpt_path, devices[0])
    sampler = DDIMSampler(model)

    print("Loading Teacher Model to CPU")
    model_orig = load_model_from_config(config_path, ckpt_path, "cpu")
    model_orig.half() 
    sampler_orig = DDIMSampler(model_orig)

    # Choose parameters to train
    parameters = []
    for name, param in model.model.diffusion_model.named_parameters():
        if train_method == 'noxattn':
            if name.startswith('out.') or 'attn2' in name or 'time_embed' in name:
                pass
            else:
                parameters.append(param)
        if train_method == 'selfattn':
            if 'attn1' in name:
                parameters.append(param)
        if train_method == 'xattn':
            if 'attn2' in name:
                parameters.append(param)
        if train_method == 'xattn_matching':
            if 'attn2' in name and ('to_q' in name or 'to_k' in name or 'to_v' in name):
                parameters.append(param)
        if train_method == 'full':
            parameters.append(param)
        if train_method == 'notime':
            if not (name.startswith('out.') or 'time_embed' in name):
                parameters.append(param)
        if train_method == 'xlayer':
            if 'attn2' in name:
                if 'output_blocks.6.' in name or 'output_blocks.8.' in name:
                    parameters.append(param)
        if train_method == 'selflayer':
            if 'attn1' in name:
                if 'input_blocks.4.' in name or 'input_blocks.7.' in name:
                    parameters.append(param)
    
    def decode_and_save_image(model_orig, z, path):
        model_orig.to(devices[0])
        z = z.to(devices[0])
        if next(model_orig.parameters()).dtype == torch.float16:
            z = z.half()
        x = model_orig.decode_first_stage(z)
        x = torch.clamp((x + 1.0)/2.0, min=0.0, max=1.0)
        x = rearrange(x, 'b c h w -> b h w c')
        image = Image.fromarray((x[0].cpu().float().numpy()*255).astype(np.uint8))
        plt.imshow(image)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(path)
        plt.close()
        model_orig.to("cpu")

    model.train()
    quick_sample_till_t = lambda cond, s, code, t: sample_model(model, sampler,
                                                                 cond, image_size, image_size, ddim_steps, s, ddim_eta,
                                                                 start_code=code, till_T=t, verbose=False)

    losses = []
    opt = torch.optim.Adam(parameters, lr=lr)
    criteria = torch.nn.MSELoss()
    history_dict = {}
    
    scaler = torch.cuda.amp.GradScaler()

    name = f'compvis-adversarial-gumbel-word_{word_print}-method_{train_method}-sg_{start_guidance}-ng_{negative_guidance}-iter_{iterations}-lr_{lr}-info_{args.info}'
    models_path = args.models_path
    os.makedirs(f'evaluation_folder/{name}', exist_ok=True)
    os.makedirs(f'invest_folder/{name}', exist_ok=True)
    os.makedirs(f'{models_path}/{name}', exist_ok=True)

    # TRAINING CODE
    pbar = tqdm(range(args.pgd_num_steps*iterations))

    def create_prompt(word, retrieve=True):
        if retrieve:
            return retrieve_embedding_token(model_name='SD-v1-4', query_token=word, vocab=args.vocab)
        else:
            prompt = f'{word}'
            emb = model.get_learned_conditioning([prompt])
            init = emb
            return init

    fixed_start_code = torch.randn((1, 4, 64, 64)).to(devices[0])    
    
    if not os.path.exists('models/embedding_matrix_dict_EN3K.pt'):
        save_embedding_matrix(model, model_name='SD-v1-4', save_mode='dict', vocab='EN3K')

    if not os.path.exists('models/embedding_matrix_array_EN3K.pt'):
        save_embedding_matrix(model, model_name='SD-v1-4', save_mode='array', vocab='EN3K')

    tokens_embedding = []
    all_sim_dict = dict()
    for word in erased_words:
        top_k_tokens, sorted_sim_dict = search_closest_tokens(word, model, k=args.gumbel_k_closest, sim='l2', model_name='SD-v1-4', ignore_special_tokens=args.ignore_special_tokens, vocab=args.vocab)
        tokens_embedding.extend(top_k_tokens)
        all_sim_dict[word] = {key:sorted_sim_dict[key] for key in top_k_tokens}

    if args.gumbel_num_centers > 0:
        assert args.gumbel_num_centers % len(erased_words) == 0, 'Number of centers should be divisible by number of erased words'
    preserved_dict = dict()

    for word in erased_words:
        temp = learn_k_means_from_input_embedding(sim_dict=all_sim_dict[word], num_centers=args.gumbel_num_centers)
        preserved_dict[word] = temp

    history_dict = save_to_dict(preserved_dict, f'preserved_set_0', history_dict)

    print('Creating preserved matrix')
    one_hot_dict = dict()
    preserved_matrix_dict = dict()
    for erase_word in erased_words:
        preserved_set = preserved_dict[erase_word]
        for i, word in enumerate(preserved_set):
            if i == 0:
                preserved_matrix = create_prompt(word)
            else:
                preserved_matrix = torch.cat((preserved_matrix, create_prompt(word)), dim=0)
        
        preserved_matrix = preserved_matrix.flatten(start_dim=1) 
        one_hot = torch.zeros((1, preserved_matrix.shape[0]), device=devices[0], dtype=preserved_matrix.dtype) 
        one_hot = one_hot + 1 / preserved_matrix.shape[0]
        one_hot = Variable(one_hot, requires_grad=True)
        one_hot_dict[erase_word] = one_hot
        preserved_matrix_dict[erase_word] = preserved_matrix
    
    print('one_hot_dict initialized')
    history_dict = save_to_dict(one_hot_dict, f'one_hot_dict_0', history_dict)

    opt_one_hot = torch.optim.Adam([one_hot for one_hot in one_hot_dict.values()], lr=args.gumbel_lr)

    def gumbel_softmax(logits, temperature=args.gumbel_temp, hard=args.gumbel_hard, eps=1e-10):
        u = torch.rand_like(logits)
        gumbel = -torch.log(-torch.log(u + eps) + eps)
        y = logits + gumbel
        y = torch.nn.functional.softmax(y / temperature, dim=-1)
        if hard != 0:
            y_hard = torch.zeros_like(logits)
            y_hard.scatter_(-1, torch.argmax(y, dim=-1, keepdim=True), 1.0)
            y = (y_hard - y).detach() + y
        return y


    for i in pbar:
        word = random.sample(erased_words,1)[0]

        opt.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)
        model_orig.zero_grad(set_to_none=True)
        opt_one_hot.zero_grad(set_to_none=True)

        prompt_0 = ''
        prompt_n = f'{word}'

        emb_0 = model.get_learned_conditioning([prompt_0])
        emb_n = model.get_learned_conditioning([prompt_n])

        emb_r = torch.reshape(torch.matmul(gumbel_softmax(one_hot_dict[word]), preserved_matrix_dict[word]).unsqueeze(0), (1, 77, 768))
        
        t_enc = torch.randint(ddim_steps, (1,), device=devices[0])
        og_num = round((int(t_enc)/ddim_steps)*1000)
        og_num_lim = round((int(t_enc+1)/ddim_steps)*1000)

        t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=devices[0])

        start_code = torch.randn((1, 4, 64, 64)).to(devices[0])

        # === 1. Generate Images with Student Model ===
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                z = quick_sample_till_t(emb_n.to(devices[0]), start_guidance, start_code, int(t_enc))
                z_r = quick_sample_till_t(emb_r.to(devices[0]), start_guidance, start_code, int(t_enc))

        # === 2. Compute Teacher Targets & Offload to CPU ===
        model_orig.to(devices[0])
        with torch.no_grad():
            z_half = z.to(devices[0]).half()
            z_r_half = z_r.to(devices[0]).half()
            t_enc_ddpm_dev = t_enc_ddpm.to(devices[0])
            emb_0_half = emb_0.to(devices[0]).half()
            emb_n_half = emb_n.to(devices[0]).half()
            emb_r_half = emb_r.to(devices[0]).half()

            e_0_org = model_orig.apply_model(z_half, t_enc_ddpm_dev, emb_0_half).float()
            e_n_org = model_orig.apply_model(z_half, t_enc_ddpm_dev, emb_n_half).float()
            e_r_org = model_orig.apply_model(z_r_half, t_enc_ddpm_dev, emb_r_half).float()

            # Pre-calculate target latents (x_0 predictions) for loss
            assert torch.all(sampler.ddim_alphas[:-1] >= sampler.ddim_alphas[1:])
            alpha_bar_t = sampler.ddim_alphas[int(t_enc)].to(devices[0])
            sqrt_one_minus_alpha = torch.sqrt(1 - alpha_bar_t)
            sqrt_alpha = torch.sqrt(alpha_bar_t)

            z_n_org_pred = (z - sqrt_one_minus_alpha * e_n_org) / sqrt_alpha
            z_0_org_pred = (z - sqrt_one_minus_alpha * e_0_org) / sqrt_alpha
            z_r_org_pred = (z_r - sqrt_one_minus_alpha * e_r_org) / sqrt_alpha
            
            # Move targets to CPU immediately
            z_n_org_pred = z_n_org_pred.cpu()
            z_0_org_pred = z_0_org_pred.cpu()
            z_r_org_pred = z_r_org_pred.cpu()

        # Clear GPU memory
        del e_0_org, e_n_org, e_r_org, z_half, z_r_half, emb_0_half, emb_n_half, emb_r_half
        model_orig.to("cpu")
        torch.cuda.empty_cache()

        # === 3. Sequential Forward/Backward to Save Memory ===
        
        # NOTE: z_n_wo_prompt_pred = (z - sqrt_one_minus_alpha * e_n_wo_prompt) / sqrt_alpha
        # Loss N depends on z_n_org_pred and z_0_org_pred
        
        if i % args.pgd_num_steps == 0:
            loss_total_item = 0
            
            # --- Part A: Erased Concept (N) ---
            with torch.cuda.amp.autocast():
                e_n_wo_prompt = model.apply_model(z.to(devices[0]), t_enc_ddpm.to(devices[0]), emb_n.to(devices[0])).float()
                z_n_wo_prompt_pred = (z - sqrt_one_minus_alpha * e_n_wo_prompt) / sqrt_alpha
                
                # Calculate loss using CPU targets moved back to GPU just for this op
                target_n = z_0_org_pred.to(devices[0]) - (negative_guidance * (z_n_org_pred.to(devices[0]) - z_0_org_pred.to(devices[0])))
                loss_n = criteria(z_n_wo_prompt_pred, target_n)

            scaler.scale(loss_n).backward()
            loss_total_item += loss_n.item()
            
            # Free Graph A
            del e_n_wo_prompt, z_n_wo_prompt_pred, loss_n, target_n
            torch.cuda.empty_cache()

            # --- Part B: Preserved Concept (R) ---
            with torch.cuda.amp.autocast():
                e_r_wo_prompt = model.apply_model(z_r.to(devices[0]), t_enc_ddpm.to(devices[0]), emb_r.to(devices[0])).float()
                z_r_wo_prompt_pred = (z_r - sqrt_one_minus_alpha * e_r_wo_prompt) / sqrt_alpha
                
                target_r = z_r_org_pred.to(devices[0])
                loss_r = criteria(z_r_wo_prompt_pred, target_r)

            scaler.scale(loss_r).backward()
            loss_total_item += loss_r.item()
            
            # Free Graph B
            del e_r_wo_prompt, z_r_wo_prompt_pred, loss_r, target_r
            
            losses.append(loss_total_item)
            pbar.set_postfix({"loss": loss_total_item})
            history_dict = save_to_dict(loss_total_item, 'loss', history_dict)
            
            scaler.step(opt)
            scaler.update()

        else:
            # Adversarial step (Maximize loss R)
            with torch.cuda.amp.autocast():
                e_r_wo_prompt = model.apply_model(z_r.to(devices[0]), t_enc_ddpm.to(devices[0]), emb_r.to(devices[0])).float()
                z_r_wo_prompt_pred = (z_r - sqrt_one_minus_alpha * e_r_wo_prompt) / sqrt_alpha
                target_r = z_r_org_pred.to(devices[0])
                loss = - criteria(z_r_wo_prompt_pred, target_r)
            
            # This is lightweight enough to do directly
            loss.backward()
            opt_one_hot.step()

        # save checkpoint and loss curve
        if i % (args.save_freq) == 0:
            with torch.no_grad():
                for word in erased_words:
                    preserved_set = preserved_dict[word]
                    word_r = preserved_set[torch.argmax(one_hot_dict[word], dim=1)]
                    emb_r_eval = torch.reshape(torch.matmul(gumbel_softmax(one_hot_dict[word]), preserved_matrix_dict[word]).unsqueeze(0), (1, 77, 768))
                    emb_n_eval = model.get_learned_conditioning([word])
                    with torch.cuda.amp.autocast():
                        z_r_till_T = quick_sample_till_t(emb_r_eval.to(devices[0]), start_guidance, fixed_start_code, int(ddim_steps))
                        z_n_till_T = quick_sample_till_t(emb_n_eval.to(devices[0]), start_guidance, fixed_start_code, int(ddim_steps))
                    decode_and_save_image(model_orig, z_r_till_T, path=f'evaluation_folder/{name}/im_r_till_T_{i}_{word}_{word_r}.png')
                    decode_and_save_image(model_orig, z_n_till_T, path=f'evaluation_folder/{name}/im_n_till_T_{i}_{word}.png')

        if i % 100 == 0:
            save_history(losses, name, word_print, models_path=models_path)
            torch.save(history_dict, f'invest_folder/{name}/history_dict_{i}.pt')

    model.eval()
    save_model(model, name, None, models_path=models_path, save_compvis=True, save_diffusers=True, compvis_config_file=config_path, diffusers_config_file=diffusers_config_path)
    save_history(losses, name, word_print, models_path=models_path)
    
def save_model(model, name, num, models_path, compvis_config_file=None, diffusers_config_file=None, device='cpu', save_compvis=True, save_diffusers=True):
    folder_path = f'{models_path}/{name}'
    os.makedirs(folder_path, exist_ok=True)
    if num is not None:
        path = f'{folder_path}/{name}-epoch_{num}.pt'
    else:
        path = f'{folder_path}/{name}.pt'

    if save_compvis:
        torch.save(model.state_dict(), path)

    if save_diffusers:
        print('Saving Model in Diffusers Format')
        savemodelDiffusers(name, compvis_config_file, diffusers_config_file, device=device )

def save_history(losses, name, word_print, models_path):
    folder_path = f'{models_path}/{name}'
    os.makedirs(folder_path, exist_ok=True)
    with open(f'{folder_path}/loss.txt', 'w') as f:
        f.writelines([str(i) for i in losses])
    plot_loss(losses,f'{folder_path}/loss.png' , word_print, n=3)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Finetuning stable diffusion model to erase concepts')
    parser.add_argument('--prompt', help='prompt corresponding to concept to erase', type=str, required=True)
    parser.add_argument('--train_method', help='method of training', type=str, required=True)
    parser.add_argument('--start_guidance', help='guidance of start image used to train', type=float, required=False, default=3)
    parser.add_argument('--negative_guidance', help='guidance of negative training used to train', type=float, required=False, default=1)
    parser.add_argument('--iterations', help='iterations used to train', type=int, required=False, default=1000)
    parser.add_argument('--lr', help='learning rate used to train', type=float, required=False, default=1e-5)
    parser.add_argument('--config_path', help='config path for stable diffusion v1-4 inference', type=str, required=False, default='configs/stable-diffusion/v1-inference.yaml')
    parser.add_argument('--ckpt_path', help='ckpt path for stable diffusion v1-4', type=str, required=False, default='models/ldm/stable-diffusion-v1/sd-v1-4-full-ema.ckpt')
    parser.add_argument('--diffusers_config_path', help='diffusers unet config json path', type=str, required=False, default='diffusers_unet_config.json')
    parser.add_argument('--devices', help='cuda devices to train on', type=str, required=False, default='0,0')
    parser.add_argument('--seperator', help='separator if you want to train bunch of erased_words separately', type=str, required=False, default=None)
    parser.add_argument('--image_size', help='image size used to train', type=int, required=False, default=512)
    parser.add_argument('--ddim_steps', help='ddim steps of inference used to train', type=int, required=False, default=50)
    parser.add_argument('--info', help='info to add to model name', type=str, required=False, default='')
    parser.add_argument('--save_freq', help='frequency to save data, per iteration', type=int, required=False, default=10)
    parser.add_argument('--models_path', help='method of prompting', type=str, required=True, default='models')

    parser.add_argument('--gumbel_lr', help='learning rate for prompt', type=float, required=False, default=1e-3)
    parser.add_argument('--gumbel_temp', help='temperature for gumbel softmax', type=float, required=False, default=2)
    parser.add_argument('--gumbel_hard', help='hard for gumbel softmax, 0: soft, 1: hard', type=int, required=False, default=0, choices=[0,1])
    parser.add_argument('--gumbel_num_centers', help='number of centers for kmeans, if <= 0 then do not apply kmeans', type=int, required=False, default=100)
    parser.add_argument('--gumbel_update', help='update frequency for preserved set, if <= 0 then do not update', type=int, required=False, default=100)
    parser.add_argument('--gumbel_time_step', help='time step for the starting point to estimate epsilon', type=int, required=False, default=0)
    parser.add_argument('--gumbel_multi_steps', help='multi steps for calculating the output', type=int, required=False, default=2)
    parser.add_argument('--gumbel_k_closest', help='number of closest tokens to consider', type=int, required=False, default=1000)
    parser.add_argument('--ignore_special_tokens', help='ignore special tokens in the embedding matrix', type=bool, required=False, default=True)
    parser.add_argument('--vocab', help='vocab', type=str, required=False, default='EN3K')
    parser.add_argument('--pgd_num_steps', help='number of step to optimize adversarial concepts', type=int, required=False, default=2)


    args = parser.parse_args()
    
    prompt = args.prompt
    train_method = args.train_method
    start_guidance = args.start_guidance
    negative_guidance = args.negative_guidance
    iterations = args.iterations
    lr = args.lr
    config_path = args.config_path
    ckpt_path = args.ckpt_path
    diffusers_config_path = args.diffusers_config_path
    devices = [f'cuda:{int(d.strip())}' for d in args.devices.split(',')]
    seperator = args.seperator
    image_size = args.image_size
    ddim_steps = args.ddim_steps

    train(prompt=prompt, train_method=train_method, start_guidance=start_guidance, negative_guidance=negative_guidance, iterations=iterations, lr=lr, config_path=config_path, ckpt_path=ckpt_path, diffusers_config_path=diffusers_config_path, devices=devices, seperator=seperator, image_size=image_size, ddim_steps=ddim_steps, args=args)