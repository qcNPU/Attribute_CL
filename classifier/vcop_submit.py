import torch
import torch.nn as nn
from torch.nn import functional as F

from tqdm import tqdm
from copy import deepcopy
import numpy as np

from clip.clip_2 import load, tokenize
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
import dataset.incremental_dataloader

from .utils import build_cosine_scheduler, cosine_loss
import time
from dataset.gpt_generation import structure

class PromptLearner(nn.Module):
    def __init__(self, args, class_names, clip_model, prompt_pool, ctx_len=12, prompt_pos=2):
        super().__init__()
        self.clip_model = clip_model
        self.args = args
        self.cls_num = len(class_names)
        self.ctx_len = ctx_len
        self.ctx_dim = clip_model.ln_final.weight.shape[0]
        self.dtype = clip_model.dtype
        self.prompt_pool = prompt_pool

        prompt_prefix =' '.join(['x'] * ctx_len * self.args.text_prompt)
        prompts = [prompt_prefix + ' ' + name + '.' for name in class_names]
        classnames = [name.replace('_', ' ') for name in class_names]
        self.name_lens = [len(_tokenizer.encode(name)) for name in class_names]
        self.prompt_pos = prompt_pos

        tokenized_prompts = torch.cat([tokenize(p) for p in prompts])
        self.tokenized_prompts = tokenized_prompts
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts.cuda()).type(self.dtype)
        self.register_buffer( 'token_prefix', embedding[:, :1, :])
        self.register_buffer( 'token_suffix', embedding[:, 1+(ctx_len * self.args.text_prompt):, :])

        nc_prompts = [prompt_prefix+'.' ]
        nc_tokenized_prompts = torch.cat([tokenize(p) for p in nc_prompts])
        self.nc_tokenized_prompts = nc_tokenized_prompts
        with torch.no_grad():
            embedding = clip_model.token_embedding(nc_tokenized_prompts.cuda()).type(self.dtype)
        self.register_buffer('nc_token_prefix', embedding[:, :1,:])
        self.register_buffer('nc_token_suffix', embedding[:, 1 + ctx_len:, :])


    def forward(self,indices_g,indices_a, test_class=None, infer=False):
        if infer:
            prompt_prefix =' '.join(['x'] * self.ctx_len * self.args.text_prompt)
            prompts = [prompt_prefix + ' ' + name + '.' for name in test_class]
            self.name_lens = [len(_tokenizer.encode(name)) for name in test_class]

            self.prompt_pos = self.prompt_pos

            tokenized_prompts = torch.cat([tokenize(p) for p in prompts])
            self.tokenized_prompts = tokenized_prompts
            with torch.no_grad():
                embedding = self.clip_model.token_embedding(tokenized_prompts.cuda()).type(self.dtype)
            self.register_buffer( 'token_prefix', embedding[:, :1, :]) # SOS, [n_cls, 1, ctx_dim]
            self.register_buffer( 'token_suffix', embedding[:, 1+(self.ctx_len * self.args.text_prompt):, :]) # CLS, EOS, [n_cls, -1, ctx_dim]
            self.cls_num = len(test_class)
        batch = indices_g.shape[0]
        prom_global = self.prompt_pool.global_prompt[indices_g]             #(32,3,12,768)
        prom_attri = self.prompt_pool.attribute_prompt[indices_a]           #(32,3,12,768)
        # ctx=self.prompt_pool[indices].view(batch, self.ctx_len * self.args.text_prompt, self.ctx_dim) #(32,36,768)
        ctx=torch.cat([prom_global,prom_attri]).view(batch, -1, self.ctx_dim)#(32,72,768)

        if self.prompt_pos == 2:
            # combined_prompts = torch.cat((global_prompt.unsqueeze(0).expand(attribute_prompts.size(0), -1), attribute_prompts), dim=1)
            prefix = self.token_prefix.unsqueeze(0).repeat(batch,1,1,1)#self.token_prefix:(10,1,768)   prefix:(32,10,1,768)
            suffix = self.token_suffix.unsqueeze(0).repeat(batch,1,1,1)#self.token_suffix:(10,40,768)  suffix:(32,10,40,768)
            ctx = ctx.unsqueeze(1).repeat(1, self.cls_num, 1, 1) #ctx:(32,10,72,768)
            prompts = torch.cat([prefix, ctx, suffix],dim=2)#(32,10,113,768)
        elif self.prompt_pos == 1:
            prompts =[]
            half_n_ctx = self.ctx_len // 2
            for i in range(self.cls_num):
                name_len = self.name_lens[i]
                prefix_i = self.token_prefix[i:i+1, :,:].unsqueeze(1)
                class_i = self.token_suffix[i:i+1,:name_len, :].unsqueeze(1)
                suffix_i = self.token_suffix[i:i+1, name_len:,:].unsqueeze(1)
                ctx_i_half1 = ctx[:,:half_n_ctx, :].unsqueeze(0)
                ctx_i_half2 = ctx[:, half_n_ctx:,:].unsqueeze(0)
                prompt = torch.cat([prefix_i, ctx_i_half1, class_i, ctx_i_half2, suffix_i],dim=2)
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        elif self.prompt_pos == 0:
            prompts =[]
            for i in range(self.cls_num):
                name_len = self.name_lens[i]
                prefix_i = self.token_prefix[i:i+1,:,:].unsqueeze(1)
                class_i = self.token_suffix[i:i+1, :name_len,:].unsqueeze(1)
                suffix_i = self.token_suffix[i:i+1, name_len:,:].unsqueeze(1)
                ctx_i = ctx.unsqueeze(0)
                prompt = torch.cat([prefix_i, class_i, ctx_i, suffix_i], dim=2)
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        prompts = prompts.squeeze(2).view(batch * self.cls_num, -1, self.ctx_dim)#(32,10,113,768)->(320,113,768)
        tokenized_prompts = self.tokenized_prompts.view(self.cls_num, -1)
        tokenized_prompts = tokenized_prompts.unsqueeze(0).repeat(batch,1,1).view(batch * self.cls_num, -1)#(320,77)
        self.prompts = prompts
        self.prompts_token = tokenized_prompts
        if infer:
            return prompts, tokenized_prompts
        else:
            nc_prompts, nc_tokenized_prompts = self.only_prefix()
            return prompts, tokenized_prompts, nc_prompts, nc_tokenized_prompts

    def only_prefix(self):
        ctx = self.prompt_pool.global_prompt
        prompt_size = ctx.shape[0]
        nc_tokenized_prompts = self.nc_tokenized_prompts.repeat(prompt_size, 1)
        prefix = self.nc_token_prefix.repeat(prompt_size, 1, 1)
        suffix = self.nc_token_suffix.repeat(prompt_size, 1, 1)
        nc_prompts = torch.cat([prefix, ctx, suffix],dim=1)
        return nc_prompts, nc_tokenized_prompts


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, x, tokenized_prompts):
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


def getTaskAttributeEmbedding(args,class_names,clip_model,text_encoder):
    entities = getTaskEntitys(args, class_names)

    with torch.no_grad():
        # 遇到DataParallel’ object has no attribute ‘xxxx’时，在model后面加上.module.
        tokenized_keys = torch.cat([tokenize(p) for p in entities]).cuda()
        entity_embeddings = clip_model.token_embedding(tokenized_keys).type(clip_model.dtype)
        entity_embeddings = text_encoder(entity_embeddings, tokenized_keys)
        entity_embeddings /= entity_embeddings.norm(dim=-1, keepdim=True)  # 归一化

    return entity_embeddings


class PromptPool:
    def __init__(self,global_key,global_prompt,attribute_key,attribute_prompt):
        self.global_key = global_key
        self.global_prompt = global_prompt
        self.attribute_key =attribute_key
        self.attribute_prompt =attribute_prompt


class CLIP(nn.Module):
    def __init__(self, args, class_names, clip_model, global_key, global_prompt, ctx_len=12):
        super().__init__()
        self.class_num = len(class_names)
        self.args = args
        self.logit_scale = clip_model.logit_scale
        self.ctx_dim = clip_model.ln_final.weight.shape[0]

        # 1. module 1：text prompt encoder
        self.text_encoder = TextEncoder(clip_model)
        self.global_key = global_key
        self.global_prompt = global_prompt
        # 1.1 将attribute送入text encoder，得到feautre，作为attribute key；
        self.attribute_key = getTaskAttributeEmbedding(args,class_names,clip_model,self.text_encoder)  # (260,768)

        if torch.cuda.device_count() > 1:
            self.text_encoder = nn.DataParallel(self.text_encoder)

        # 3. 初始化对应数量的attribute prompt   ,n_ctx：上下文长度，ctx_dim：上下文维度 n_cls：类别数量 name_lens：每个类别名称的长度
        attribute_prompt = torch.empty(self.attribute_key.shape[0], ctx_len, self.ctx_dim, dtype=clip_model.dtype).cuda()
        nn.init.normal_(attribute_prompt, std=0.02)
        attribute_prompt = nn.Parameter(attribute_prompt)
        self.attribute_prompt = attribute_prompt

        # 2. module 2：text template enoder
        self.template_encoder = TextEncoder(clip_model)

        self.prompt_pool = PromptPool(self.global_key,self.global_prompt,self.attribute_key,self.attribute_prompt)
        # 3. module 3：prompt learner
        self.prompt_learner = PromptLearner(self.args, class_names, clip_model, self.prompt_pool, ctx_len=ctx_len)

        # 4. module 4：image encoder
        self.image_encoder = clip_model.visual

    def forward(self, image, num_test=None, test_class=None, test=False):

        with torch.no_grad():
            image_features = self.image_encoder(image.type(self.dtype))#image:(32,3,32,32)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            image_features = image_features.detach()

        if not test:
            # combined_keys = torch.cat((self.global_key, self.attribute_keys), dim=1)
            probability_g = image_features @ self.global_key.t()    #(32,768)  (10,768)
            _, indices_g = probability_g.topk(k=min(self.args.text_prompt, probability_g.shape[1]), dim=1, largest=True)#(32,3)
            key_choose_g = self.global_key[indices_g]#(32,3,768)

            probability_a = image_features @ self.attribute_key.t()
            _, indices_a = probability_a.topk(k=min(self.args.text_prompt, probability_a.shape[1]), dim=1, largest=True)
            key_choose_a = self.attribute_key[indices_a]

            text_prompt, tokenized_prompts, nc_prompts, nc_tokenized_prompts = self.prompt_learner(indices_g,indices_a)
            text_features = self.text_encoder(text_prompt, tokenized_prompts) # (320,113,768)  (320,77)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.view(image_features.shape[0], self.class_num, -1)
            image_features = image_features.unsqueeze(1)
            logit_scale = self.logit_scale.exp()
            logits = logit_scale * (image_features * text_features).sum(-1)

            nc_text_features = self.text_encoder(nc_prompts, nc_tokenized_prompts)
            nc_text_features = nc_text_features / nc_text_features.norm(dim=-1, keepdim=True)
            dis = nc_text_features @ nc_text_features.permute(1, 0)
            loss_m = dis[~torch.eye(self.args.num_prompt, dtype=torch.bool, device='cuda')].abs().mean()

            key_choose = torch.cat(key_choose_g,key_choose_a)

            return logits, image_features, key_choose, loss_m
        else:
            n_test = len(test_class)
            probability_g = image_features @ self.text_key.t()
            _, indices_g = probability_g.topk(k=min(self.args.text_prompt,probability_g.shape[1]), dim=1, largest=True)

            text_prompt, tokenized_prompts = self.prompt_learner(indices_g,test_class,test)
            text_features = self.text_encoder(text_prompt,tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logit_scale = self.logit_scale.exp()
            text_features = text_features.view(image_features.shape[0], n_test, -1)
            image_features = image_features.unsqueeze(1)
            logit_scale = self.logit_scale.exp()
            logits = logit_scale * (image_features * text_features).sum(-1)
            return logits


    @property
    def dtype(self):
        return self.image_encoder.conv1.weight.dtype


class CoOp:
    def __init__(self, prev_key, prev_prompt, args, use_float32=False, use_grad_checkpoint=False, keep=False):
        clip_model, _ = load(args.ckpt_path)
        clip_model.eval()
        if use_float32:
            clip_model.float()
        self.clip_model = clip_model
        self.use_grad_checkpoint = use_grad_checkpoint
        self.num_prompt = args.num_prompt
        self.ctx_len = args.ctx_len
        self.lr = args.lr*args.train_batch/20
        self.wd = args.wd
        self.epochs = args.epochs
        self.train_batch = args.train_batch 
        self.args = args
        dtype = clip_model.dtype
        self.dtype = dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        # 1. 初始化prompt pool
        # 1.1 global key：（keyNum, ctx_dim）——（key数量 ，每个vector为多少维）
        text_key = torch.empty(self.num_prompt, ctx_dim, dtype=self.dtype).cuda()
        nn.init.normal_(text_key, std=0.02)
        # 1.2 global prompt：（keyNum, ctx_len, ctx_dim）——（key数量，每个prompt包含几个learnable vector，每个vector为多少维）
        text_prompt = torch.empty(self.num_prompt, self.ctx_len, ctx_dim, dtype=self.dtype).cuda()
        nn.init.normal_(text_prompt, std=0.02)

        if  keep == True :
            self.text_key = nn.Parameter(prev_key)
            self.text_prompt = nn.Parameter(prev_prompt)
        else:
            self.text_key = nn.Parameter(text_key)
            self.text_prompt = nn.Parameter(text_prompt)




    def fit(self, data, len_train):

        train_loader = data['train_loader']
        ima_proto = {}
        for n in range(self.args.class_per_task):
            ima_proto[int(n)] = []

        if len(train_loader.dataset)< self.train_batch:
            real_img_bsz = len(train_loader.dataset)
            self.lr = self.lr * real_img_bsz / self.train_batch 
        else:
            real_img_bsz = self.train_batch

        per_epoch_steps = len(train_loader)

        self.init_model(class_names=data['class_names'], per_epoch_steps=per_epoch_steps,text_key=self.text_key, text_prompt=self.text_prompt)

        self.model.eval()

        for epoch in range(self.epochs):
            loop = tqdm(train_loader, total=len(train_loader))
            loop.set_description(f'Epoch [{epoch}/{self.epochs}]')
            for idx, (x, y) in enumerate(loop):
                
                y = y - self.args.class_per_task * self.args.sess
                lab_idx = y.cpu().numpy().tolist()
                cur_iter_idx = epoch*per_epoch_steps+idx
                self.cur_iter_idx = cur_iter_idx
                self.scheduler.step(cur_iter_idx)

                output, ima_feat, key_choose, loss_m = self.model(x.cuda())
                
                loss_main = F.cross_entropy(output, y.cuda())
                loss_k = cosine_loss(ima_feat,key_choose)
                loss = loss_main + 0.7*loss_k + 0.3*loss_m
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()



    def init_model(self, class_names, per_epoch_steps, text_key, text_prompt):

        self.n_class = len(class_names)
        clip_model = deepcopy(self.clip_model)

        self.model = CLIP(self.args, class_names, clip_model, text_key, text_prompt, self.ctx_len)
        if self.use_grad_checkpoint:
            try:
                self.model.text_encoder.transformer.use_gradient_checkpoint = True 
            except:
                self.model.text_encoder.module.transformer.use_gradient_checkpoint = True

        Other_params = [param for name, param in self.model.named_parameters() if 'text_key' in name]
        param_dict = [{'params': [p for p in self.model.prompt_learner.parameters() if p.requires_grad]}, 
                        {'params': Other_params}]

        self.optimizer = torch.optim.SGD(param_dict, lr=self.lr, weight_decay=self.wd)
        self.scheduler = build_cosine_scheduler(
            self.optimizer,
            lr=self.lr,
            total_step=self.epochs*per_epoch_steps)

    @torch.no_grad()
    def accuracy(self, loader, num_test, test_class, mean_per_class=False):
        if mean_per_class:
            return self._accuracy_mpc(loader, num_test, test_class)
        else:
            #因为每个class的test数量是一样的，所以data-level的平均和task-level的结果一样
            # print("taskMean acc",self._accuracy_mean_task(loader, num_test, test_class))
            return self._accuracy(loader, num_test, test_class)

    def _accuracy_mpc(self, loader, num_test, test_class):
        n_class = self.n_class
        acc_per_class = [0 for _ in range(n_class)]
        count_per_class = [0 for _ in range(n_class)]
        for i, (x, y) in enumerate(loader):
            pred_y = self.inference(x.cuda())
            _, top_labels = pred_y.topk(1, dim=-1)
            for c in range(n_class):
                acc_per_class[c] += ((top_labels.view(-1) == y.cuda()) * (y.cuda()== c)).sum().item()
                count_per_class[c] += (y.cuda() == c).sum().item()
        acc = [a*1.0/c for (a, c) in zip(acc_per_class, count_per_class)]
        acc = np.array(acc).mean()
        return acc

    def _accuracy(self, loader, ses, test_class):
        total_count=0
        acc_count =0

        taskclassMap = {}
        for t in range(ses):
            taskclassMap[t]= list(range(t*self.args.class_per_task,(t+1)*self.args.class_per_task))
        for i,(x, y) in enumerate(loader):
            pred_y = self.inference(x.cuda(), ses, test_class)
            _, top_labels = pred_y.topk(1, dim=-1)
            acc_count += (top_labels.view(-1)==y.cuda()).sum().cpu().numpy()

            total_count += y.shape[0]
        acc = acc_count*1.0/total_count
        acc = acc.item()
        # print("data-level,match={},total={}".format(acc_count,total_count))
        return acc


    def _accuracy_mean_task(self, loader, ses, test_class):
        n_class = self.n_class
        acc_per_task = [0 for _ in range(ses+1)]
        count_per_task = [0 for _ in range(ses+1)]
        for i, (x, y) in enumerate(loader):
            pred_y = self.inference(x.cuda(),ses, test_class)
            _, top_labels = pred_y.topk(1, dim=-1)
            for t in range(ses+1):
                acc_per_task[t] += ((top_labels.view(-1) == y.cuda()) * (y.cuda()// self.args.class_per_task== t)).sum().item()
                count_per_task[t] += (y.cuda()// self.args.class_per_task == t).sum().item()
        acc = [a*1.0/c for (a, c) in zip(acc_per_task, count_per_task)]
        acc = np.array(acc).mean()
        # print("task-level,match={},total={}".format(str(acc_per_task), str(count_per_task)))
        return acc


    @torch.no_grad()
    def inference(self,image, num_test, test_class):
        logits = self.model(image, num_test, test_class, test=True)
        return logits.float().softmax(dim=-1)

def getTaskEntitys(args, train_classnames):
    # 取出task中所有class的entity和attribute，合并去重
    class_structures = structure.get_Classes_Structures(args, train_classnames)
    entities = []
    for classname, info in class_structures.items():
        for j in info:
            entities.extend(j["Entities"] + j["Attributes"])
    keys = set(i.lower() for i in entities)
    return keys
