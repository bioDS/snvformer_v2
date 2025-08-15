from trainer import *
import os
from encoder import CustomEncoder
import h5py
from torch.backends.cuda import sdp_kernel
from packaging import version
from contextlib import nullcontext

import importlib
import trainer
import vectorizer
import encoder
importlib.reload(trainer)
importlib.reload(vectorizer)
importlib.reload(encoder)
from trainer import *
from vectorizer import *
from encoder import *

def get_model(args):
    num_snv_classes = get_num_snv_classes(args.h5_file)
    use_phenos = ["age", "sex", "bmi"]
    encoder = custom_encoder_from_args(args, use_phenos)
    return encoder


def main(args):
    ddp_setup()

    model = get_model(args)
    dataset, mini_test_set, verify_set, optimizer, warmup_scheduler, main_scheduler = load_train_objs(args, model)
    use_devices = [int(item) for item in args.devices.split(',')]
    print(f"using devices: {use_devices}")
    if os.environ["RANK"] == "0":
        num_params = sum(p.numel() for p in model.parameters())
        print(f"model has {num_params} parameters")
    train_data = prepare_dataloader(dataset, args.batch_size)
    mini_test_data = prepare_dataloader(mini_test_set, args.batch_size)
    verify_data = prepare_dataloader(verify_set, args.batch_size)
    trainer = EncoderTrainer(model, train_data, mini_test_data, verify_data, optimizer, warmup_scheduler, main_scheduler, args.save_every, args.snapshot_path, use_devices, args, args.limit_steps, args.gradual_length, args.mask_frac)

    has_sdp = version.parse(torch.__version__) >= version.parse("2.0")
    with sdp_kernel(**backend_map[SDPBackend.FLASH_ATTENTION]) if (has_sdp and not args.no_gradscaler) else nullcontext():
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(not args.no_gradscaler)):
            trainer.train(args.total_epochs)

    destroy_process_group()


def interactive_setup():
    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"


if __name__ == "__main__":
    parser = get_default_argparser()
    parser.add_argument("--snapshot-path", default="/data/ukbb/v2_snapshots/encoder.pt")
    args = parser.parse_args()
    if args.embed_dim == 0:
        args.embed_dim = args.snv_embed_size + args.gene_embed_size 
        if args.ignore_chrom == False:
            args.embed_dim += args.chrom_embed_size
        if args.pos_combine == "cat":
            args.embed_dim += args.pos_embed_size
        if args.ignore_class == False:
            args.embed_dim += len(vectorizer.InputTypes)
        if args.embed_dim % args.num_heads != 0:
            additional_snv = args.num_heads - (args.embed_dim % args.num_heads)
            args.snv_embed_size += additional_snv
            args.embed_dim += additional_snv

    main(args)

def testing():
    interactive_setup()
    ddp_setup()

    parser = get_default_argparser()
    parser.add_argument("--snapshot-path", default="/data/ukbb/v2_snapshots/encoder.pt")
    args = parser.parse_args()
    args.encoder_type="linformer"
    args.total_epochs=1
    args.num_layers=4
    args.batch_size=6
    args.embed_dim=64
    args.num_heads=4
    args.seq_len=65803
    args.linformer_k=64
    args.h5_file="/data/ukbb/net_input/genotyped_p1e-1.h5"
    args.test_frac=0.3
    args.devices="0"
    args.snapshot_path="/data/ukbb/v2_snapshots/snvformer_encoder.pt"
    args.warmup_batches=500
    args.warmup_min_lr=1e-7
    args.warmup_max_lr=1e-7
    args.main_scheduler_step_size=1000
    args.main_scheduler_gamma=1.00
    args.report_on_batch=1000
    args.no_gradscaler = True
    args.position_encoding="fixed"
    args.snv_encoding="embedding"
    args.ignore_chrom = True
    args.pos_combine="cat"
    args.torch_sdp = False
    args.ffn_scale=1
    args.dropout=0.05
    args.snv_embed_size=8
    args.pos_embed_size=56
    args.ignore_class = True
    args.tf_init = False

    model = get_model(args)
    dataset, mini_test_set, verify_set, optimizer, warmup_scheduler, main_scheduler = load_train_objs(args, model)
    use_devices = [int(item) for item in args.devices.split(',')]
    print(f"using devices: {use_devices}")
    if os.environ["RANK"] == "0":
        num_params = sum(p.numel() for p in model.parameters())
        print(f"model has {num_params} parameters")
    train_data = prepare_dataloader(dataset, args.batch_size)
    mini_test_data = prepare_dataloader(mini_test_set, args.batch_size)
    verify_data = prepare_dataloader(verify_set, args.batch_size)
    trainer = EncoderTrainer(model, train_data, mini_test_data, verify_data, optimizer, warmup_scheduler, main_scheduler, args.save_every, args.snapshot_path, use_devices, args, args.limit_steps, args.gradual_length, args.mask_frac)

    trainer.train(args.total_epochs)
