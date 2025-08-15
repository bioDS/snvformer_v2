from trainer import *
import os
from encoder import CustomEncoder, ContinuousTransformerPredictor, LinearModel, custom_encoder_from_args
from packaging import version
from contextlib import nullcontext


def get_model(args):
    if args.model_type == "transformer":
        encoder = custom_encoder_from_args(args, use_phenos=["sex"])
        snapshot = torch.load(args.encoder_snapshot)
        encoder.load_state_dict(snapshot["MODEL_STATE"])
        model = ContinuousTransformerPredictor(encoder, output_transform_layers=args.output_transform_layers)
    else:
        model = LinearModel(args.seq_len, 1)
    return model


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
    trainer = HeightTrainer(model, train_data, mini_test_data, verify_data, optimizer, warmup_scheduler, main_scheduler, args.save_every, args.snapshot_path, use_devices, args, args.limit_steps)
    has_sdp = version.parse(torch.__version__) >= version.parse("2.0")

    if (args.model_type == "transformer"):
        with sdp_kernel(**backend_map[SDPBackend.FLASH_ATTENTION]) if (has_sdp and not args.no_gradscaler) else nullcontext():
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(not args.no_gradscaler)):
                trainer.train(args.total_epochs)
    else:
        trainer.train(args.total_epochs)
    destroy_process_group()


def interactive_setup():
    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"


if __name__ == "__main__":
    parser = get_default_argparser()
    parser.add_argument("--snapshot-path", default="/data/ukbb/v2_snapshots/height_predictor.pt")
    parser.add_argument("--model-type", default="transformer", help="\'transformer\' or \'linear\'")
    args = parser.parse_args()
    
    main(args)