import os.path as osp
import torch
import torch.nn as nn
from open_clip.tokenizer import tokenize as tokenize


class PromptLearner(nn.Module):
    def __init__(self, n_ctx, model):
        super().__init__()
        self.dtype = model.logit_scale.dtype

        ctx_init = "a video of action "
        ctx_init = ctx_init.replace("_", " ")
        n_ctx = len(ctx_init.split(" "))
        prompt = tokenize(ctx_init)
        with torch.no_grad():
            embedding = model.text.token_embedding(prompt).type(self.dtype)
        ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]

        self.prompt_prefix = ctx_init
        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        self.n_ctx = n_ctx

    def forward(self, classnames, model):

        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [
            self.prompt_prefix + " " + name + "." for name in classnames
        ]  # ['X X X X X X X X classname.']
        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(
            "cuda"
        )  # (1, 77)
        with torch.no_grad():
            embedding = model.text.token_embedding(tokenized_prompts).type(
                self.dtype
            )  # (1, 77, 768)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer(
            "token_suffix", embedding[:, 1 + self.n_ctx :, :]
        )  # CLS, EOS

        self.n_cls = 1
        self.n_ctx = self.n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,  # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        return prompts


class TextEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.transformer = model.transformer
        self.positional_embedding = model.positional_embedding
        self.ln_final = model.ln_final
        self.text_projection = model.text_projection
        self.dtype = self.ln_final.weight.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = (
            x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)]
            @ self.text_projection
        )

        return x
