import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F

from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class PromptLearner(nn.Module):
    def __init__(self, classnamesDict, clip_model):
        super().__init__()
        classnamesDict = {key: [label.replace('a photo of a ', '') for label in classnamesDict[key]] for key in classnamesDict.keys()}
        classnamesDict['obj'] = ['person']
        self.classnamesDict = classnamesDict

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # use given words to initialize context vectors
        ctx_init = "a photo of a"
        n_ctx = len(ctx_init.split(" "))
        prompt = clip.tokenize(ctx_init)
        with torch.no_grad():
            embedding = clip_model.token_embedding(prompt.to(device)).type(clip_model.dtype)
        ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
        prompt_prefix = ctx_init

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        token_prefix = {}
        token_suffix = {}
        tokenized_promptsDict = {}
        for key in classnamesDict.keys():
            classnames = classnamesDict[key]
            prompts = [prompt_prefix + " " + name + "." for name in classnames]
            tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
            with torch.no_grad():
                embedding = clip_model.token_embedding(tokenized_prompts.to(device)).type(clip_model.dtype)
            token_prefix[key] = embedding[:, :1, :]
            token_suffix[key] = embedding[:, 1 + n_ctx :, :]
            tokenized_promptsDict[key] = tokenized_prompts

        self.token_prefix = token_prefix
        self.token_suffix = token_suffix
        # self.register_buffer("token_prefix", token_prefix)  # SOS
        # self.register_buffer("token_suffix", token_suffix)  # CLS, EOS
        self.tokenized_promptsDict = tokenized_promptsDict  # torch.Tensor

    def forward(self):
        prefix = self.token_prefix
        suffix = self.token_suffix

        promptsDict = {}
        for key in self.classnamesDict.keys():
            classnames = self.classnamesDict[key]
            ctx = self.ctx
            ctx = ctx.unsqueeze(0).expand(len(classnames), -1, -1)

            prompts = torch.cat(
                [
                    prefix[key],  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix[key],  # (n_cls, *, dim)
                ],
                dim=1,
            )
            promptsDict[key] = prompts

        return promptsDict

# class CustomCLIP(nn.Module):
#     def __init__(self, cfg, classnames, clip_model):
#         super().__init__()
#         self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
#         self.tokenized_prompts = self.prompt_learner.tokenized_prompts
#         self.image_encoder = clip_model.visual
#         self.text_encoder = TextEncoder(clip_model)
#         self.logit_scale = clip_model.logit_scale
#         self.dtype = clip_model.dtype

#     def forward(self, image):
#         image_features = self.image_encoder(image.type(self.dtype))

#         prompts = self.prompt_learner()
#         tokenized_prompts = self.tokenized_prompts
#         text_features = self.text_encoder(prompts, tokenized_prompts)

#         image_features = image_features / image_features.norm(dim=-1, keepdim=True)
#         text_features = text_features / text_features.norm(dim=-1, keepdim=True)

#         logit_scale = self.logit_scale.exp()
#         logits = logit_scale * image_features @ text_features.t()

#         return logits

# @TRAINER_REGISTRY.register()
# class CoOp(TrainerX):
#     def build_model(self):
#         cfg = self.cfg
#         classnames = self.dm.dataset.classnames

#         model, _ = clip.load(model_name)
#         self.model = CustomCLIP(cfg, classnames, clip_model)
#         for name, param in self.model.named_parameters():
#             if "prompt_learner" not in name:
#                 param.requires_grad_(False)

#         self.model.to(self.device)


#         self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

#         # Note that multi-gpu training could be slow because CLIP's size is
#         # big, which slows down the copy operation in DataParallel
#         device_count = torch.cuda.device_count()
#         if device_count > 1:
#             print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
#             self.model = nn.DataParallel(self.model)

#     def forward_backward(self, batch):
#         image, label = self.parse_batch_train(batch)
        
#         prec = self.cfg.TRAINER.COOP.PREC
#         if prec == "amp":
#             with autocast():
#                 output = self.model(image)
#                 loss = F.cross_entropy(output, label)
#             self.optim.zero_grad()
#             self.scaler.scale(loss).backward()
#             self.scaler.step(self.optim)
#             self.scaler.update()
#         else:
#             output = self.model(image)
#             loss = F.cross_entropy(output, label)
#             self.model_backward_and_update(loss)

#         loss_summary = {
#             "loss": loss.item(),
#             "acc": compute_accuracy(output, label)[0].item(),
#         }

#         if (self.batch_idx + 1) == self.num_batches:
#             self.update_lr()

#         return loss_summary

#     def parse_batch_train(self, batch):
#         input = batch["img"]
#         label = batch["label"]
#         input = input.to(self.device)
#         label = label.to(self.device)
#         return input, label

#     def load_model(self, directory, epoch=None):
#         if not directory:
#             print("Note that load_model() is skipped as no pretrained model is given")
#             return

#         names = self.get_model_names()

#         # By default, the best model is loaded
#         model_file = "model-best.pth.tar"

#         if epoch is not None:
#             model_file = "model.pth.tar-" + str(epoch)

#         for name in names:
#             model_path = osp.join(directory, name, model_file)

#             if not osp.exists(model_path):
#                 raise FileNotFoundError('Model not found at "{}"'.format(model_path))

#             checkpoint = load_checkpoint(model_path)
#             state_dict = checkpoint["state_dict"]
#             epoch = checkpoint["epoch"]

#             # Ignore fixed token vectors
#             if "token_prefix" in state_dict:
#                 del state_dict["token_prefix"]

#             if "token_suffix" in state_dict:
#                 del state_dict["token_suffix"]

#             print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
#             # set strict=False
#             self._models[name].load_state_dict(state_dict, strict=False)


