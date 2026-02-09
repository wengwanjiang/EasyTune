import torch, pickle, random
from torch import nn
import os
from typing import List, Optional, Union
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions.distribution import Distribution

import datetime

from .utils import lengths_to_mask, mld_collate_motion_only
from .opt.attention import (
    SkipTransformerEncoder,
    SkipTransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
)
from .opt.position_encoding import build_position_encoding
from .opt.embeddings import TimestepEmbedding, Timesteps

class KLLoss:
    def __call__(self, q, p):
        mu_q, logvar_q = q
        mu_p, logvar_p = p

        log_var_ratio = logvar_q - logvar_p
        t1 = (mu_p - mu_q).pow(2) / logvar_p.exp()
        div = 0.5 * (log_var_ratio.exp() + t1 - 1 - log_var_ratio)
        return div.mean()

    def __repr__(self):
        return "KLLoss()"

class InfoNCE_with_filtering:
    def __init__(self, temperature=0.7, threshold_selfsim=0.8):
        self.temperature = temperature
        self.threshold_selfsim = threshold_selfsim
        self.a, self.b = 0, 1
    def get_sim_matrix(self, x, y):
        x_logits = torch.nn.functional.normalize(x, dim=-1)
        y_logits = torch.nn.functional.normalize(y, dim=-1)
        sim_matrix = x_logits @ y_logits.T
        return sim_matrix

    def __call__(self, x, y, sent_emb=None):
        bs, device = len(x), x.device
        sim_matrix = self.get_sim_matrix(x, y) / self.temperature

        if sent_emb is not None and self.threshold_selfsim:
            # put the threshold value between -1 and 1
            real_threshold_selfsim = 2 * self.threshold_selfsim - 1
            # Filtering too close values
            # mask them by putting -inf in the sim_matrix
            #selfsim = sent_emb @ sent_emb.T
            sent_emb_normalized = torch.nn.functional.normalize(sent_emb, dim=-1)
            selfsim = sent_emb_normalized @ sent_emb_normalized.T
            
            selfsim_nodiag = selfsim - selfsim.diag().diag()
            idx = torch.where(selfsim_nodiag > real_threshold_selfsim)
            #true_count = torch.sum(selfsim_nodiag > real_threshold_selfsim).item()
            ## 计算比例
            #total_elements = selfsim_nodiag.numel()
            #true_ratio = self.a / self.b 
            #self.a += true_count
            #self.b += total_elements
#
            ## 打印结果
            #print(f"True ratio: {true_ratio:.4f}")
            #print('Sim Ratio = ', torch.sum(selfsim_nodiag > real_threshold_selfsim).item()/64/64)
            sim_matrix[idx] = -torch.inf

        labels = torch.arange(bs, device=device)

        total_loss = (
            F.cross_entropy(sim_matrix, labels) + F.cross_entropy(sim_matrix.T, labels)
        ) / 2

        return total_loss

    def __repr__(self):
        return f"Constrastive(temp={self.temp})"
def process_T5_outputs(raw_texts, T5_model):
    with torch.no_grad():
        T5_outputs = T5_model.encode(raw_texts, show_progress_bar=False, batch_size=len(raw_texts),output_value=None)
        #input_ids = torch.stack([item['input_ids'] for item in T5_outputs])
        attn_masks = torch.stack([item['attention_mask'] for item in T5_outputs])
        token_embeddings = torch.stack([item['token_embeddings'] for item in T5_outputs])
        sentence_embeddings = torch.stack([item['sentence_embedding'] for item in T5_outputs])
        t_length = attn_masks.sum(1).tolist()
        return t_length, token_embeddings, sentence_embeddings.unsqueeze(1)
class SPM(nn.Module):

    def __init__(self,
                 #cfg, 
                 #ablation,
                 nfeats: int = 263,
                 latent_dim: list = [1, 256],
                 ff_size: int = 1024,
                 num_layers: int = 9,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 arch: str = "encoder_decoder",
                 normalize_before: bool = False,
                 activation: str = "gelu",
                 position_embedding: str = "learned",
                 temp: float = 0.1,
                 thr: float = 0.9,
                 ckpt_path: str = None,
                 mld_path: str = None,
                 lambda_t2m=0, 
                 lambda_m2m=0,
                 spm_mode=None,
                 clip_path=None,
                 **kwargs) -> None:

        super().__init__()
        self.clip = None
        if clip_path is not None:# or True:
            from sentence_transformers import SentenceTransformer
            self.clip = SentenceTransformer(clip_path)                
            #self.clip = SentenceTransformer("sentence-transformers/sentence-t5-large", cache_folder='/data/wwj/llm')

        self.latent_size = latent_dim[0]
        self.latent_dim = latent_dim[-1]

        self.arch = arch
        self.mlp_dist = 'mld' #ablation.MLP_DIST
        self.pe_type = 'mld' #ablation.PE_TYPE

        
        self.query_motion_pos_encoder = build_position_encoding(
            self.latent_dim, position_embedding=position_embedding)
        self.query_text_pos_encoder = build_position_encoding(
            self.latent_dim, position_embedding=position_embedding)
        self.query_pos_decoder = build_position_encoding(
            self.latent_dim, position_embedding=position_embedding)


        encoder_layer = TransformerEncoderLayer(self.latent_dim, num_heads, ff_size,
                                            dropout, activation, normalize_before,)
        encoder_norm = nn.LayerNorm(self.latent_dim)
        self.motion_encoder = SkipTransformerEncoder(encoder_layer, num_layers,
                                              encoder_norm)
        self.text_encoder = SkipTransformerEncoder(encoder_layer, num_layers,
                                              encoder_norm)
        if self.arch == "all_encoder":
            pass
        elif self.arch == "encoder_decoder":
            decoder_layer = TransformerDecoderLayer(
                self.latent_dim,
                num_heads,
                ff_size,
                dropout,
                activation,
                normalize_before,
            )
            decoder_norm = nn.LayerNorm(self.latent_dim)
            self.decoder = SkipTransformerDecoder(decoder_layer, num_layers,
                                                  decoder_norm)


        if self.mlp_dist:
            self.global_motion_token = nn.Parameter(torch.randn(self.latent_size, self.latent_dim))
            self.dist_motion_layer = nn.Linear(self.latent_dim, 2 * self.latent_dim)

            self.global_text_token = nn.Parameter(torch.randn(self.latent_size, self.latent_dim))
            self.dist_text_layer = nn.Linear(self.latent_dim, 2 * self.latent_dim)
        else:
            pass
        self.skel_embedding = nn.Linear(nfeats, self.latent_dim)
        self.text_embedding = nn.Linear(1024, self.latent_dim)
        self.final_layer = nn.Linear(self.latent_dim, nfeats)
        self.time_proj = Timesteps(512, True, 0)
        self.time_embedding = TimestepEmbedding(512, 256)

        self.recons_loss_fn = torch.nn.SmoothL1Loss(reduction="mean")
        self.latent_loss_fn = torch.nn.SmoothL1Loss(reduction="mean")
        self.kl_loss_fn = KLLoss()
        self.cl_loss_fn = InfoNCE_with_filtering(temperature=temp, threshold_selfsim=thr)
        print(f'self.cl_loss_fn = InfoNCE_with_filtering(temperature={temp}, threshold_selfsim={thr})')
        print('================Construct SPM Start================')
        #ckpt_path = '/data/wwj/ckpt/SPM_New/TFS_SPM_Pured_1e-4_BS128_T1_M0T0_E50.pth'
        self.spm_mode = 'M1T0' if 'M1T0' in ckpt_path else 'M0T0'
        print(f'Init SPM with {ckpt_path}, {self.spm_mode}')
        self.lambda_t2m = lambda_t2m
        self.lambda_m2m = lambda_m2m
        print(f'(LAMBDA_T2M, LAMDA_M2M) = ({self.lambda_t2m}, {self.lambda_m2m}), Query Repo = [./GradGuidance/KD_BIKL_CEH3D_20_Repo_woIO.pkl]')
        #self.query_repo = pickle.load(open('./GradGuidance/KD_BIKL_CEH3D_20_Repo_woIO.pkl', 'rb'))
        #self.repo_Mean = self.query_repo['Mean']
        #self.repo_Std = self.query_repo['Std']
        #del self.query_repo['Mean'], self.query_repo['Std']
        print('================Construct SPM Eendd================')
        #exit(0)
        self.step_cnt = 0
    

    #lmd:
#  recons: 1.0
#  latent: 1.0e-5
#  kl: 1.0e-5
#  contrastive: 0.1
#
#lr: 1e-4
#temperature: 0.1
#threshold_selfsim: 0.80
#threshold_selfsim_metrics: 0.95
    def compute_loss(self, m_rst, t_rst, m_ref, m_dist, t_dist, m_latent, t_latent, sent_emb):#, joints_ref, joints_rst_m, joints_rst_t, t_time_embed, m_time_embed):
        recons_loss = self.recons_loss_fn(t_rst, m_ref) + + self.recons_loss_fn(m_rst, m_ref)# + self.recons_loss_fn(m_rst, t_rst) 
        #recons_loss += self.recons_loss_fn(joints_rst_m, joints_ref) + self.recons_loss_fn(joints_rst_t, joints_ref)# + self.recons_loss_fn(joints_rst_t, joints_rst_m)
        #recons_loss += self.recons_loss_fn(t_time_embed, m_time_embed) * 0.1
        m_dist = [m_dist.loc, 2 * torch.log(m_dist.scale)]# decode (mu, logvar) from torch.distributions.Normal(mu, std)
        t_dist = [t_dist.loc, 2 * torch.log(t_dist.scale)]# decode (mu, logvar) from torch.distributions.Normal(mu, std)
        ref_mus = torch.zeros_like(m_dist[0])
        ref_logvar = torch.zeros_like(t_dist[1])
        ref_dist = (ref_mus, ref_logvar)
        kl_loss = (
            self.kl_loss_fn(t_dist, m_dist)  # text_to_motion
            + self.kl_loss_fn(m_dist, t_dist)  # motion_to_text
            + self.kl_loss_fn(m_dist, ref_dist)  # motion
            + self.kl_loss_fn(t_dist, ref_dist)  # text
        )
        latent_loss = self.latent_loss_fn(m_latent, t_latent)
        cl_loss = self.cl_loss_fn(t_latent.squeeze(), m_latent.squeeze(), sent_emb)
        #return recons_loss*1e0 + kl_loss*1e-5 + latent_loss*1e-4 + cl_loss*1e-1
        return recons_loss*1e0 + kl_loss*1e-5 + latent_loss*1e-5 + cl_loss*1e-1
    def forward(self, motion_feature, text: Tensor, m_len: Optional[List[int]] = None, timestep=None, mode='M0T0'):
        with torch.no_grad():
            t_len, sent_emb, cls_token = process_T5_outputs(text, self.clip)
        if mode == 'M1T0':
            t_latent, t_dist, t_time_embed = self.encode_text(sent_emb, t_len, return_time=True)
            m_latent, m_dist, m_time_embed = self.encode_motion(motion_feature, m_len, timestep=timestep, return_time=True)
        elif mode == 'M0T1':
            t_latent, t_dist = self.encode_text(sent_emb, t_len, timestep=timestep)
            m_latent, m_dist = self.encode_motion(motion_feature, m_len)
        elif mode == 'M1T1':
            t_latent, t_dist, t_time_embed = self.encode_text(sent_emb, t_len, timestep=timestep, return_time=True)
            m_latent, m_dist, m_time_embed = self.encode_motion(motion_feature, m_len, timestep=timestep, return_time=True)
        elif mode == 'M0T0':
            t_latent, t_dist = self.encode_text(sent_emb, t_len)
            m_latent, m_dist = self.encode_motion(motion_feature, m_len)
        m_rst = self.decode(m_latent, m_len) # recons from motion 
        t_rst = self.decode(t_latent, m_len) # recons from text this length used to order generated motion length
        #joints_ref = 0 # self.feats2joints(motion_feature)
        #joints_rst_m = 0 # self.feats2joints(m_rst)
        #joints_rst_t = 0 # self.feats2joints(t_rst)
        loss = self.compute_loss(m_rst, t_rst, motion_feature, m_dist, t_dist, m_latent, t_latent, cls_token.squeeze())
                                  #joints_ref, joints_rst_m, joints_rst_t, t_time_embed, m_time_embed)
        #loss += self.time_cl_loss(timestep, m_time_embed.squeeze())
        return loss #, m_latent
    # def time_cl_loss(self, timestep, time_token):
    #     eps = 1e-8
    #     temperature = 0.1
    #     time_token = F.normalize(time_token, p=2, dim=1)  # L2归一化
    #     similarity = torch.matmul(time_token, time_token.T)  # [N, N]
    #     N = similarity.size(0)
    #     # 生成正样本对的mask（相同时间步）
    #     mask = torch.eq(timestep.unsqueeze(1), timestep.unsqueeze(0)).float()  # [N, N]
    #     mask = mask - torch.eye(N, device=mask.device)  # 排除自身
    #     # 计算正样本相似度（平均）
    #     pos_sim = torch.sum(similarity * mask, dim=1)  # [N]
    #     pos_count = torch.sum(mask, dim=1) + eps  # 避免除零
    #     pos_loss = -torch.log(pos_sim / pos_count / temperature + eps)
        
    #     # 计算负样本相似度（logsumexp）
    #     neg_sim = similarity / temperature
    #     neg_loss = torch.logsumexp(neg_sim, dim=1)  # [N]
        
    #     # 总损失
    #     loss = (pos_loss + neg_loss).mean()
    #     return loss * 1e-2

    def get_reward_t2m(self, raw_texts, motion_feats, t_len=None, sent_emb=None, m_len=None, timestep=None, return_m=False):
        #with torch.no_grad():
        #    t_len, sent_emb, _ = self.clip(raw_texts)
        with torch.enable_grad():
            t_latent, _ = self.encode_text(sent_emb, t_len, timestep=None)
            if self.spm_mode == 'M1T0':
                m_latent, _ = self.encode_motion(motion_feats, m_len, timestep=None)#timestep)
            elif self.spm_mode == 'M0T0':
                m_latent, _ = self.encode_motion(motion_feats, m_len, timestep=None)
            
            reward = torch.nn.functional.cosine_similarity(t_latent.squeeze(), m_latent.squeeze(), dim=-1) # .mean()
            #grad, = torch.autograd.grad(outputs=reward, inputs=xt, create_graph=False, retain_graph=False)
        # with open('./log_final/DRTune-10-5-50.log', 'a') as f:
        with open('./lsx.log', 'a') as f:
            data_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            print(f"{data_str} Step: {self.step_cnt} T2M Reward: {reward}\n")
            # f.write(f"{data_str} Step: {self.step_cnt} T2M Reward: {reward.item()}\n")
            self.step_cnt += 1
        if return_m:
            return reward, m_latent  #grad
        return reward

#    @torch.enable_grad()
#    def get_reward_m2m(self, closest_motion, closest_m_len, motion_feats, m_len, t=None):
#        closest_motion = torch.cat([closest_motion, closest_motion], dim=0)
#        if self.spm_mode[1] == '1':
#            m_latent_gt, _ = self.encode_motion(closest_motion, closest_m_len,t//20)
#            m_latent_fake, _ = self.encode_motion(motion_feats, m_len, t)
#        elif self.spm_mode[1] == '0':
#            m_latent_gt, _ = self.encode_motion(closest_motion, closest_m_len)
#            m_latent_fake, _ = self.encode_motion(motion_feats, m_len)
#        similarity = torch.nn.functional.cosine_similarity(m_latent_gt.squeeze(), m_latent_fake.squeeze(), dim=-1).mean()
#        return similarity
    
    def encode_motion(
            self,
            features: Tensor,
            lengths: Optional[List[int]] = None,
            timestep=None,
            return_time=False
    ) -> Union[Tensor, Distribution]:

        if lengths is None:
            lengths = [len(feature) for feature in features]
        # For Motion Stream, feature = [32, 40 ,263] => [bs, nframes, nfeats], lengths = [32], argmax=40
        
        device = features.device

        bs, nframes, nfeats = features.shape
        mask = lengths_to_mask(lengths, device, max_len=nframes)

        x = features
        # Embed each human poses into latent vectors
        x = self.skel_embedding(x) # 263 --> 256

        # Switch sequence and batch_size because the input of
        # Pytorch Transformer is [Sequence, Batch size, ...]
        x = x.permute(1, 0, 2)  # now it is [nframes, bs, latent_dim]

        # Each batch has its own set of tokens
        dist = torch.tile(self.global_motion_token[:, None, :], (1, bs, 1)) # [2, 32, 256]

        # create a bigger mask, to allow attend to emb
        dist_masks = torch.ones((bs, dist.shape[0]), # [2, 32, 256]
                                dtype=bool,
                                device=x.device)
        # adding the embedding token for all sequences 
        if timestep is not None:
            timesteps = timestep.expand(x.shape[1]).clone()
            time_emb = self.time_proj(timesteps)
            time_emb = time_emb.to(dtype=x.dtype)
            time_emb = self.time_embedding(time_emb).unsqueeze(0)
            xseq = torch.cat((dist, time_emb, x), 0) # [42, 32, 256]
            time_masks = torch.ones((bs,time_emb.shape[0]), # [2, 32, 256]
                                dtype=bool,
                                device=x.device)
            aug_mask = torch.cat((dist_masks, time_masks, mask), 1) # [32, 40+2]
        else:
            xseq = torch.cat((dist, x), 0) # [42, 32, 256] # 1,1,256 [1, bs, 256] [45, 1, 256][len, bs, 256]
            aug_mask = torch.cat((dist_masks, mask), 1) # [32, 40+2]

        xseq = self.query_motion_pos_encoder(xseq)
        dist = self.motion_encoder(xseq,
                                src_key_padding_mask=~aug_mask)[:dist.shape[0]]
        time_emb = xseq[dist.shape[0]]
        # content distribution
        # self.latent_dim => 2*self.latent_dim
        if self.mlp_dist:
            tokens_dist = self.dist_motion_layer(dist)
            mu = tokens_dist[:, :, :self.latent_dim]
            logvar = tokens_dist[:, :, self.latent_dim:]
        else:
            mu = dist[0:self.latent_size, ...] # [1 32 256]
            logvar = dist[self.latent_size:, ...] # [1 32 256]
        #print('motion', mu, logvar)
        # resampling
        std = logvar.exp().pow(0.5)
        dist = torch.distributions.Normal(mu, std)
        latent = dist.rsample()
        if return_time:
            return latent, dist, time_emb
        return latent, dist

    def decode(self, z: Tensor, lengths: List[int]):
        mask = lengths_to_mask(lengths, z.device)
        bs, nframes = mask.shape

        queries = torch.zeros(nframes, bs, self.latent_dim, device=z.device, requires_grad=True)

        # todo
        # investigate the motion middle error!!!

        # Pass through the transformer decoder
        # with the latent vector for memory
        if self.arch == "all_encoder":
            xseq = torch.cat((z, queries), axis=0)
            z_mask = torch.ones((bs, self.latent_size), dtype=bool, device=z.device)
            augmask = torch.cat((z_mask, mask), axis=1)
            xseq = self.query_pos_decoder(xseq)
            output = self.decoder(xseq, src_key_padding_mask=~augmask)[z.shape[0]:]


        elif self.arch == "encoder_decoder":
            queries = self.query_pos_decoder(queries)
            output = self.decoder(tgt=queries, memory=z, tgt_key_padding_mask=~mask).squeeze(0)


        output = self.final_layer(output) # 40 32 256 --> 40 32 263
        # zero for padded area
        output[~mask.T] = 0
        # Pytorch Transformer: [Sequence, Batch size, ...]
        feats = output.permute(1, 0, 2)
        return feats
    def encode_text(
            self,
            features: Tensor, # features = self.clip(texts) # BS N 768
            lengths: Optional[List[int]] = None,
            timestep = None,
            return_time=False
    ) -> Union[Tensor, Distribution]:
        #if lengths is None:
        #    lengths = [len(feature) for feature in features]
        # For Motion Stream, feature = [32, 40 ,263] => [bs, nbtokens, nfeats], lengths = [32], argmax=40
        
        device = features.device

        bs, nbtokens, nfeats = features.shape
        mask = lengths_to_mask(lengths, device, max_len=nbtokens)

        x = features
        # Embed each human poses into latent vectors
        
        x = self.text_embedding(x) # 768 --> 256

        # Switch sequence and batch_size because the input of
        # Pytorch Transformer is [Sequence, Batch size, ...]
        x = x.permute(1, 0, 2)  # now it is [nbtokens, bs, latent_dim]

        # Each batch has its own set of tokens
        dist = torch.tile(self.global_text_token[:, None, :], (1, bs, 1)) # [2, 32, 256]

        # create a bigger mask, to allow attend to emb
        dist_masks = torch.ones((bs, dist.shape[0]), # [2, 32, 256]
                                dtype=bool,
                                device=x.device)
        
        # adding the embedding token for all sequences 
        if timestep is not None:
            timesteps = timestep.expand(x.shape[1]).clone()
            time_emb = self.time_proj(timesteps)
            time_emb = time_emb.to(dtype=x.dtype)
            time_emb = self.time_embedding(time_emb).unsqueeze(0)
            xseq = torch.cat((dist, time_emb, x), 0) # [42, 32, 256]
            time_masks = torch.ones((bs,time_emb.shape[0]), # [2, 32, 256]
                                dtype=bool,
                                device=x.device)
            aug_mask = torch.cat((dist_masks, time_masks, mask), 1) # [32, 40+2]
        else:
            xseq = torch.cat((dist, x), 0) # [42, 32, 256] # 1,1,256 [1, bs, 256] [45, 1, 256][len, bs, 256]
            aug_mask = torch.cat((dist_masks, mask), 1) # [32, 40+2]

        
        xseq = self.query_text_pos_encoder(xseq)
        dist = self.text_encoder(xseq,
                            src_key_padding_mask=~aug_mask)[:dist.shape[0]]
        time_emb = xseq[dist.shape[0]]
        # content distribution
        # self.latent_dim => 2*self.latent_dim
        if self.mlp_dist:
            tokens_dist = self.dist_text_layer(dist)
            mu = tokens_dist[:, :, :self.latent_dim]
            logvar = tokens_dist[:, :, self.latent_dim:]
        else:
            mu = dist[0:self.latent_size, ...] # [1 32 256]
            logvar = dist[self.latent_size:, ...] # [1 32 256]
        #print('text', mu, logvar)
        # resampling
        std = logvar.exp().pow(0.5)
        dist = torch.distributions.Normal(mu, std)
        latent = dist.rsample()
        if return_time:
            return latent, dist, time_emb
        return latent, dist
    

    def query_closest(self, query_texts):
        batch = []
        #self.repo_Mean
        #self.repo_Std
        bs = len(query_texts)
        for text in query_texts:
            if len(text) < 3:
                continue
            # cloest_text = self.query_repo[text]['top_repo_texts'][0]
            # cloest_npy = np.load(self.query_repo[text]['top_repo_belongs'][0])
            # cloest_npy = np.load(self.query_repo[text]['top_repo_belongs'][0])
            motion = self.query_repo[text]['top_repo_npy'][0]
            orig_len = len(motion)
            coin2 = np.random.choice(["single", "single", "double"])
            if coin2 == "double":
                target_len = (orig_len // 4 - 1) * 4
            else:  # coin2 == "single"
                target_len = (orig_len // 4) * 4
            idx = random.randint(0, orig_len - target_len)
            motion = motion[idx: idx + target_len]
            # 如果后续需要 m_len 与裁剪后 motion 的长度保持一致，就用 target_len 或重新赋值：
            m_len = len(motion)
            motion = (motion - self.repo_Mean) / self.repo_Std

            batch.append([motion, m_len])   
        if len(batch) == bs // 2:  
            batch = batch + batch
        assert len(batch) == bs, 'Error'
        return mld_collate_motion_only(batch)
