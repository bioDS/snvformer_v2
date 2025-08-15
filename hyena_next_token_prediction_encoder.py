from trainer import *
import os
import h5py
from standalone_hyenadna import HyenaDNAModel
from hyena_encoder import get_model

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
    trainer = HyenaNextTokenEncoderTrainer(
                                args.view_size,
                                model,
                                train_data,
                                mini_test_data,
                                verify_data,
                                optimizer,
                                warmup_scheduler,
                                main_scheduler,
                                args.save_every,
                                args.snapshot_path,
                                use_devices,
                                args,
                                args.limit_steps,
                                args.gradual_length,
                                args.mask_frac,
    )

    # with sdp_kernel(**backend_map[SDPBackend.FLASH_ATTENTION]) if (has_sdp and not args.no_gradscaler) else nullcontext():
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        trainer.train(args.total_epochs)
    destroy_process_group()


def interactive_setup():
    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--total-epochs', default=2, type=int, help='Total epochs to train the model')
    parser.add_argument('--save-every', type=int, default=1, help='How often to save a snapshot')
    parser.add_argument('--batch-size', default=32, type=int, help='Input batch size on each device (default: 32)')
    # parser.add_argument("--embed-dim", type=int, default="64")
    parser.add_argument("--num-layers", type=int, default="2") # 2 - 8 used in paper
    parser.add_argument("--num-heads", type=int, default="8")
    parser.add_argument("--seq-len", type=int, default="320") # must be updated to match input file
    parser.add_argument("--test-frac", type=float, default="0.3")
    parser.add_argument("--h5-file", default="/data/ukbb/net_input/gwas_ldprune_320.h5")
    parser.add_argument('-d', '--devices', help='comma delimited list of gpus to use', type=str, default="4")
    parser.add_argument("--limit-steps", type=int, default="-1")
    # parser.add_argument("--encoder-snapshot", default="/data/ukbb/v2_snapshots/hyena_encoder.pt")
    parser.add_argument("--augment-data", action='store_true')
    parser.add_argument("--warmup-min-lr", type=float, default="1e-5")
    parser.add_argument("--warmup-max-lr", type=float, default="1e-2")
    parser.add_argument("--warmup-batches", type=int, default="500")
    parser.add_argument("--main-scheduler-step-size", type=int, default="100")
    parser.add_argument("--main-scheduler-gamma", type=float, default="0.9")
    parser.add_argument("--verbose-scheduler", action='store_true')
    parser.add_argument("--report-on-batch", type=int, default="100")
    parser.add_argument("--no-gradscaler", action='store_true')
    parser.add_argument("--dropout", type=float, default="0.2")
    parser.add_argument("--augment-frac", type=float, default="0.15", help="fraction of snv sequence to mask/change when augmenting gout data")
    parser.add_argument("--augment-mult", type=float, default="2.0", help="ratio of augmented to actual gout data")
    # parser.add_argument("--encoder-type", default="classic", help="\'classic\' or \'linformer\'")
    # parser.add_argument("--output-transform-layers", type=int, default="5")
    parser.add_argument("--snapshot-path", default="/data/ukbb/v2_snapshots/hyena_encoder.pt")
    parser.add_argument("--d-model", type=int, default = "128") # 128 and 256 used in HyenaDNA paper
    parser.add_argument("--pretrained-hyena", action='store_const', const=None)
    parser.add_argument("--gradual-length", action='store_true')
    parser.add_argument("--mask-frac", type=float, default = "0.5") # 128 and 256 used in HyenaDNA paper
    parser.add_argument("--view-size", type=int, default=64)
    args = parser.parse_args()
    
    main(args)