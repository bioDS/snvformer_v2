import torch
import torch.nn.functional as F
from torch import nn
from vectorizer import Vectorizer
from standalone_hyenadna import create_mixer_cls, create_mlp_cls, create_block
from functools import partial
import math
from trainer import get_num_snv_classes
import argparse

# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def tf_init_weights(module, initializer_range=0.02):
    """Initialize the weights."""
    if isinstance(module, nn.Linear):
        # Slightly different from the TF version which uses truncated_normal for initialization
        # cf https://github.com/pytorch/pytorch/pull/5617
        module.weight.data.normal_(mean=0.0, std=initializer_range)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=initializer_range)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

class HyenaEncoder(nn.Module):
    def __init__(self, seq_len, embed_dim, num_heads, num_layers, dropout=0.0, device='cpu', dtype=torch.float32):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        # self.process_group = process_group
        # note max_position_embeddings is 0 for Hyena, and therefore isn't used
        # self.embeddings = GPT2Embeddings(d_model, vocab_size, max_position_embeddings,
        #                                      **factory_kwargs)
        layer_cfg = {
            "_name_": "hyena",
            "l_max": seq_len,
            "order": 3,
            "filter_order": 64,
            "num_heads": num_heads,
            "inner_factor": 1,
            "num_blocks": 1,
            "fused_bias_fc": False,
            "outer_mixing": False,
            "dropout": dropout ,
            "filter_dropout": 0.0,
            "filter_cls": 'hyena-filter',
            "post_order_ffn": False,
            "jit_filter": False,
            "short_filter_order": 3,
            "activation": "id",
            "filter_args": {
                "emb_dim": embed_dim,
                "order": 16,
                "seq_len": seq_len,
            },
            "d_model": embed_dim
        }

        residual_in_fp32=True
        self.residual_in_fp32 = residual_in_fp32
        d_model = embed_dim
        d_inner = embed_dim*4
        attn_cfg = None

        self.layers = nn.ModuleList([create_block(
            d_model, d_inner=d_inner,
            layer=layer_cfg, attn_layer_idx=None,
            attn_cfg=attn_cfg, layer_norm_epsilon=1e-5,
            resid_dropout1=dropout,
            resid_dropout2=dropout, residual_in_fp32=residual_in_fp32,layer_idx=i,
            **factory_kwargs,
        ) for i in range(num_layers)])

        self.drop_f = nn.Dropout(dropout)
        self.ln_f = nn.LayerNorm(d_model, eps=1e-5, **factory_kwargs)

        self.apply(partial(self._init_weights, n_layer=num_layers))

        
    def _init_weights(self, module, n_layer, initializer_range=0.02, rescale_prenorm_residual=True,
                    glu_act=False):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=initializer_range)

        if rescale_prenorm_residual:
            # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
            #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
            #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
            #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
            #
            # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
            for name, p in module.named_parameters():
                if name in ["out_proj.weight", "fc2.weight"]:
                    # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                    nn.init.normal_(p, mean=0.0, std=initializer_range / math.sqrt(2 * n_layer))
                # If using GLU activation for now, we scale the std by 2
                elif name in ["output_linear.0.weight"]:
                    # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                    if not glu_act:
                        nn.init.normal_(p, mean=0.0, std=initializer_range / math.sqrt(2 * n_layer))
                    else:
                        out_features = p.shape[0]
                        # Multiplying the first half of the matrix by 2 since sigmoid scales it down by 0.5
                        # on average.
                        nn.init.normal_(p[:out_features // 2], mean=0.0, std=initializer_range / math.sqrt(2 * n_layer) * 2)


    # N.B. mask is not implemented
    def forward(self, X, src_key_padding_mask=None):
        hidden_states = X
        residual = None

        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual)

        dropped = self.drop_f(hidden_states)
        residual = (dropped + residual) if residual is not None else dropped
        hidden_states = self.ln_f(residual.to(dtype=self.ln_f.weight.dtype))

        return hidden_states

def masked_softmax(X, valid_lens):
    """Perform softmax operation by masking elements on the last axis."""
    # `X`: 3D tensor, `valid_lens`: 1D or 2D tensor
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                              value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)

class D2LDotProductAttention(nn.Module):
    """Scaled dot product attention."""
    def __init__(self, dropout, use_sparsemax=False, **kwargs):
        super(D2LDotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.use_sparsemax = use_sparsemax

    # Shape of `queries`: (`batch_size`, no. of queries, `d`)
    # Shape of `keys`: (`batch_size`, no. of key-value pairs, `d`)
    # Shape of `values`: (`batch_size`, no. of key-value pairs, value
    # dimension)
    # Shape of `valid_lens`: (`batch_size`,) or (`batch_size`, no. of queries)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # Set `transpose_b=True` to swap the last two dimensions of `keys`
        # print("query shape {}".format(queries.shape))
        # print("keys  shape {}".format(keys.transpose(1,2).shape))
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
        # print("scores shape: {}".format(scores.shape))
        if (self.use_sparsemax):
            self.attention_weights = masked_sparsemax(scores, valid_lens)
        else:
            self.attention_weights = masked_softmax(scores, valid_lens)
        self.attention_weights.requires_grad_()
        self.attention_weights.retain_grad()
        return torch.bmm(self.dropout(self.attention_weights), values)

class LinformerEncoderLayer(nn.Module):
    def __init__(self, seq_len, nhead, num_hiddens, linformer_k, dropout=0.0, ffn_act = "gelu", use_torch_sdp=True, ffn_scale=4):
        super().__init__()
        ffn_hiddens = ffn_scale*num_hiddens
        self.W_q = nn.Linear(num_hiddens, num_hiddens, bias=False)
        self.W_k = nn.Linear(num_hiddens, num_hiddens, bias=False)
        self.W_v = nn.Linear(num_hiddens, num_hiddens, bias=False)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=False)
        self.E_i = nn.Linear(seq_len, linformer_k, bias=True)
        self.F_i = nn.Linear(seq_len, linformer_k, bias=True)
        self.num_heads = nhead
        self.layer_norm = nn.LayerNorm(num_hiddens)
        self.ffn_dense_1 = nn.Linear(num_hiddens,ffn_hiddens)
        self.ffn_dense_2 = nn.Linear(ffn_hiddens,num_hiddens)
        self.ffn_use_gelu = ffn_act ==  "gelu"
        if ffn_act == "gelu":
            self.ffn_act = nn.GELU()
        else:
            self.ffn_act = nn.ReLU()
        self.dropout = dropout
        self.use_torch_sdp = use_torch_sdp
        if not self.use_torch_sdp:
            self.attention = D2LDotProductAttention(dropout, False)

    # @torch.compile
    def transpose_qkv(self, X):
        """Transposition for parallel computation of multiple attention heads.
        Borrowed from d2l.ai"""
        # Shape of input X: (batch_size, no. of queries or key-value pairs,
        # num_hiddens). Shape of output X: (batch_size, no. of queries or
        # key-value pairs, num_heads, num_hiddens / num_heads)
        X = X.reshape(X.shape[0], X.shape[1], self.num_heads, -1)
        X = X.permute(0, 2, 1, 3)
        if self.use_torch_sdp:
            # Shape of output X: (batch_size, num_heads, no. of queries or key-value
            # pairs, num_hiddens / num_heads)
            return X
        else:
            ## Shape of output: (batch_size * num_heads, no. of queries or key-value
            ## pairs, num_hiddens / num_heads)
            return X.reshape(-1, X.shape[2], X.shape[3])

    # @torch.compile
    def transpose_output(self, X):
        """Reverse the operation of transpose_qkv."""
        if not self.use_torch_sdp:
            X = X.reshape(-1, self.num_heads, X.shape[1], X.shape[2])
        X = X.permute(0, 2, 1, 3)
        return X.reshape(X.shape[0], X.shape[1], -1)

    # @torch.compile
    def positionwise_ffn(self, X):
        o1 = self.ffn_dense_1(X)
        o2 = self.ffn_act(o1)
        # return self.ffn_dense_2(self.ffn_act(self.ffn_dense_1(X)))
        return self.ffn_dense_2(o2)

    # @torch.compile
    def addnorm(self, X, prev_X):
        return self.layer_norm(prev_X + X)

    def forward(self, X, src_mask, is_causal, src_key_padding_mask):
        # print(f"input: {torch.sum(X)}")
        queries = self.transpose_qkv(self.W_q(X))
        keys = self.transpose_qkv(self.W_k(X))
        values = self.transpose_qkv(self.W_v(X))

        # Linformer approximation
        if self.use_torch_sdp:
            keys = self.E_i(keys.swapaxes(2,3)).swapaxes(2,3)
            values = self.F_i(values.swapaxes(2,3)).swapaxes(2,3)
        else:
            keys = self.E_i(keys.swapaxes(1,2)).swapaxes(1,2)
            values = self.F_i(values.swapaxes(1,2)).swapaxes(1,2)
        # queries = queries.to(torch.float16)
        # keys = keys.to(torch.float16)
        # values = values.to(torch.float16)
        # print(f"queries: {torch.sum(queries)}")
        # print(f"keys: {torch.sum(keys)}")
        # print(f"values: {torch.sum(values)}")
        if self.use_torch_sdp:
            output = F.scaled_dot_product_attention(
                queries,
                keys,
                values,
                attn_mask=None,
                dropout_p=self.dropout,
                is_causal=is_causal,
            )
        else:
            output = self.attention(queries, keys, values, None)
        output = output.to(torch.float)
        # attn_output
        output_concat = self.transpose_output(output)
        # print(f"output_concat: {torch.sum(output_concat)}")
        at = self.W_o(output_concat)
        # addnorm attn_output w/ input
        ra_out = self.addnorm(at, X)
        residual2 = ra_out
        ffn_out = self.positionwise_ffn(ra_out)
        tmp = self.addnorm(ffn_out, residual2)
        # print(f"returning from transformer block: {torch.sum(tmp)}")
        return tmp

def custom_encoder_from_args_dict(args, use_phenos):
    num_snv_classes = get_num_snv_classes(args["h5_file"])
    encoder = CustomEncoder(args["embed_dim"], args["num_heads"],
            args["num_layers"], num_snv_classes, args["seq_len"],
            k=args["linformer_k"], encoder_type=args["encoder_type"], dropout=args["dropout"], use_phenos=use_phenos,
            position_encoding=args["position_encoding"], snv_encoding=args["snv_encoding"], pos_combine=args["pos_combine"],
            ignore_chrom=args["ignore_chrom"], torch_sdp=args["torch_sdp"], ffn_scale=args["ffn_scale"],
            snv_embed_size=args["snv_embed_size"], gene_embed_size=args["gene_embed_size"], use_gene_embed_graph=args["gene_embed_graph"], chrom_embed_size=args["chrom_embed_size"], pos_embed_size=args["pos_embed_size"],
            ignore_class=args["ignore_class"], tf_init=args["tf_init"])
    return encoder


def custom_encoder_from_args(args, use_phenos):
    if type(args) == argparse.Namespace:
        args = vars(args)
    return custom_encoder_from_args_dict(args, use_phenos)

class CustomEncoder(nn.Module):
    def __init__(self,
        embed_dim, num_heads, num_layers, num_snv_classes, seq_len, k=None, encoder_type="classic", dropout=0.2,
        use_phenos = ['age','sex','bmi'], position_encoding="embedding", snv_encoding="embedding", ignore_chrom=False,
        pos_combine = "add", torch_sdp=True, ffn_scale=4, snv_embed_size=None, gene_embed_size=None, use_gene_embed_graph=False, chrom_embed_size=None, pos_embed_size=None,
        ignore_class = False, tf_init=True,
    ):
        super(CustomEncoder, self).__init__()
        self.full_seq_len = seq_len+len(use_phenos)+1
        if encoder_type == "linformer":
            # encoder_layer = LinformerEncoderLayer(seq_len, k)
            encoder_layer = LinformerEncoderLayer(self.full_seq_len, num_heads, embed_dim, k, dropout, use_torch_sdp=torch_sdp, ffn_scale=ffn_scale)
        elif encoder_type == "classic":
            encoder_layer = nn.TransformerEncoderLayer(d_model = embed_dim,
                                                    nhead=num_heads,
                                                    batch_first=True,
                                                    dropout=dropout,
                                                    activation="gelu")
        if encoder_type == "linformer" or encoder_type == "classic":
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        elif encoder_type == "hyena":
            self.encoder = HyenaEncoder(seq_len=self.full_seq_len, embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, num_layers=num_layers)
        else:
            raise Exception("unrecognized encoder type")
        self.vectorizer = Vectorizer(seq_len, embed_dim, num_snv_classes, use_phenos, position_encoding, snv_encoding, pos_combine, ignore_chrom, ignore_class, dropout, snv_embed_size, gene_embed_size, use_gene_embed_graph, chrom_embed_size, pos_embed_size)
        self.encoder_type = encoder_type
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        # initialise all weights
        if (tf_init):
            self.apply(tf_init_weights)

    def forward(self, phenos, positions, chromosomes, snvs, max_len):
        if max_len == 0:
            attn_mask = None
        else:
            attn_mask = torch.cat([torch.ones(max_len, dtype=bool), torch.zeros(self.full_seq_len-max_len, dtype=bool)])
            attn_mask = attn_mask.broadcast_to((snvs.shape[0], -1)).to(snvs.device)
        X = self.vectorizer(phenos, positions, chromosomes, snvs, max_len)
        Xp = F.pad(X, (0,0,0,self.full_seq_len-X.shape[1]))
        # if (self.encoder_type == "classic"):
        #     Xp = Xp.to(torch.float16)
        # with sdp_kernel(**backend_map[SDPBackend.FLASH_ATTENTION]) if (has_sdp and not args.no_gradscaler) else nullcontext():
        # with sdp_kernel(**backend_map[SDPBackend.FLASH_ATTENTION]) if (has_sdp and not args.no_gradscaler) else nullcontext():
        #     # with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(not args.no_gradscaler)):
        #         tmp = self.encoder(Xp, src_key_padding_mask=attn_mask)
        tmp = self.encoder(Xp, src_key_padding_mask=attn_mask)
        return tmp
    
    def get_attn(self, phenos, positions, chromosomes, snvs, max_len):
        if max_len == 0:
            attn_mask = None
        else:
            attn_mask = torch.cat([torch.ones(max_len, dtype=bool), torch.zeros(self.full_seq_len-max_len, dtype=bool)])
            attn_mask = attn_mask.broadcast_to((snvs.shape[0], -1)).to(snvs.device)
        X = self.vectorizer(phenos, positions, chromosomes, snvs, max_len)
        Xp = F.pad(X, (0,0,0,self.full_seq_len-X.shape[1]))
        # tmp = self.encoder(Xp, src_key_padding_mask=attn_mask)
        tmp = self.encoder.layers[0](Xp, src_mask=None, src_key_padding_mask=attn_mask, is_causal=False)
        weights = torch.einsum("ijk->j", tmp)
        abs_weights = torch.abs(torch.einsum("ijk->j", tmp))
        return weights, abs_weights

class SimpleClassifier(nn.Module):
    def __init__(self, encoder, embed_dim):
        super().__init__()
        self.encoder = encoder
        self.classification_block = nn.Linear(embed_dim, 1)

    def forward(self, phenos, positions, chromosomes, snvs, max_len):
        X = self.encoder(phenos, positions, chromosomes, snvs, max_len)
        class_tok = X[:,0,:]
        class_out = self.classification_block(class_tok)
        return class_out

class OldDualClassifier(nn.Module):
    def __init__(self, encoder, embed_dim):
        super().__init__()
        self.encoder = encoder
        self.classification_block = nn.Sequential(
            nn.Linear(embed_dim, 2),
            nn.GELU(),
            nn.Softmax(1)
        )

    def forward(self, phenos, positions, chromosomes, snvs, max_len):
        X = self.encoder(phenos, positions, chromosomes, snvs, max_len)
        class_tok = X[:,0,:]
        class_out = self.classification_block(class_tok)
        # print(f"final output: {class_out}")
        return class_out

class FlatOutputClassifier(nn.Module):
    def __init__(self, encoder, embed_dim):
        super().__init__()
        self.encoder = encoder
        self.classification_block = nn.Linear(embed_dim*encoder.full_seq_len, 1)

    def forward(self, phenos, positions, chromosomes, snvs, max_len):
        X = self.encoder(phenos, positions, chromosomes, snvs, max_len)
        flattened_output = X.view(X.shape[0], -1) # batch_size, -1
        class_out = self.classification_block(flattened_output)
        return class_out


class SimpleHyenaClassifier(nn.Module):
    def __init__(self, encoder, embed_dim):
        super().__init__()
        self.encoder = encoder
        self.classification_block = nn.Linear(embed_dim, 2)

    def forward(self, snvs, positions):
        X = self.encoder(snvs, positions)
        class_tok = X[:,0,:]
        class_out = self.classification_block(class_tok)
        return class_out


class Classifier(nn.Module):
    def __init__(self, encoder:CustomEncoder, dropout=0.2, output_transform_layers=5):
        super().__init__()
        self.encoder = encoder
        self.classification_block = nn.Sequential(
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model = encoder.embed_dim,
                    nhead=encoder.num_heads,
                    batch_first=True,
                    dropout=dropout,
                    activation="gelu"),
                num_layers=output_transform_layers),
            nn.Linear(encoder.embed_dim, 2)
        )

    def forward(self, phenos, positions, chromosomes, snvs, max_len):
        X = self.encoder(phenos, positions, chromosomes, snvs)
        class_tok = X[:,0,:]
        class_out = self.classification_block(class_tok)
        return class_out

class ContinuousTransformerPredictor(nn.Module):
    def __init__(self, encoder:CustomEncoder, dropout=0.2, output_transform_layers=5):
        super().__init__()
        self.encoder = encoder
        self.classification_block = nn.Sequential(
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model = encoder.embed_dim,
                    nhead=encoder.num_heads,
                    batch_first=True,
                    dropout=dropout,
                    activation="gelu"),
                num_layers=output_transform_layers),
            nn.Linear(encoder.embed_dim, 1)
        )

    def forward(self, phenos, positions, chromosomes, snvs, max_len):
        X = self.encoder(phenos, positions, chromosomes, snvs, max_len)
        class_tok = X[:,0,:]
        class_out = self.classification_block(class_tok)
        return class_out

class CombinedOutputTransformer(nn.Module):
    """
    Returns both a (binary) class and a linear predictor
    intented to be (gout, urate)
    """
    def __init__(self, encoder:CustomEncoder, dropout=0.2):
        super().__init__()
        self.encoder = encoder
        embed_dim = encoder.embed_dim
        self.classification_block = nn.Linear(embed_dim, 2)
        self.linear_pred_block = nn.Linear(embed_dim, 1)

    def forward(self, phenos, positions, chromosomes, snvs, max_len):
        X = self.encoder(phenos, positions, chromosomes, snvs, max_len)
        class_tok = X[:,0,:]
        class_out = self.classification_block(class_tok)
        linear_out = self.linear_pred_block(class_tok)
        return class_out, linear_out


class MLP(nn.Module):
    def __init__(self, snv_seq_len, internal_dim, depth, dropout=0.2):
        super(MLP, self).__init__()
        self.first_layer = nn.Sequential(
            nn.Linear(snv_seq_len, internal_dim),
            nn.Dropout1d(dropout),
            nn.GELU(),
        )
        all_layers = []
        for i in range(depth-1):
            new_layer = nn.Sequential(
                nn.Linear(internal_dim, internal_dim),
                nn.Dropout1d(dropout),
                nn.GELU(),
            )
            all_layers.append(new_layer)
        self.repeat_layers = nn.ModuleList(all_layers)
        self.final_layer = nn.Linear(internal_dim,1)

    def forward(self, phenos, positions, chromosomes, snvs, max_len):
        snvs = snvs.float()
        x = self.first_layer(snvs)
        for layer in self.repeat_layers:
            x = layer(x)
        x = self.final_layer(x)
        return x

class LinearModel(nn.Module):
    def __init__(self, snv_seq_len, output_dim):
        super().__init__()
        self.final_layer = nn.Linear(snv_seq_len, output_dim)

    def forward(self, phenos, positions, chromosomes, snvs, max_len):
        snvs = snvs.float()
        x = self.final_layer(snvs)
        return x

def test_encoder():
    from streaming_dataset import StreamingDataset
    dataset = StreamingDataset("/data/ukbb/net_input/all_gwas.h5")
    phenos, positions, chromosomes, snvs = dataset[0:2]
    ce = CustomEncoder(64, 8, 2, 3, 13290)
    tmp = ce(phenos, positions, chromosomes, snvs)
    assert(tmp.shape == (2, 13290, 64))

def test_linformer_encoder():
    from streaming_dataset import StreamingDataset
    dataset = StreamingDataset("/data/ukbb/net_input/all_gwas.h5")
    phenos, positions, chromosomes, snvs = dataset[0:2]
    ce = CustomEncoder(64, 8, 2, 3, 13290, 64, encoder_type="linformer")
    tmp = ce(phenos, positions, chromosomes, snvs)
    assert(tmp.shape == (2, 13290, 64))


def test_linformer():
    seq_len = 256
    query = torch.rand(32, 8, seq_len, 64, dtype=torch.float32)
    key = query
    value = query
    tmp = F.scaled_dot_product_attention(query, key, value)
    k = 64
    lf = LinformerEncoderLayer(seq_len, k)
    tmp2 = lf(query, key, value)
    assert(tmp.shape == tmp2.shape)
