from trainer import *
import os
from encoder import CustomEncoder, ContinuousTransformerPredictor, LinearModel, custom_encoder_from_args
from packaging import version
from contextlib import nullcontext


def get_model(args):
    if args.model_type == "transformer":
        use_phenos = ["age", "sex", "bmi"]
        encoder = custom_encoder_from_args(args, use_phenos)
        snapshot = torch.load(args.encoder_snapshot)
        encoder.load_state_dict(snapshot["MODEL_STATE"])
        model = ContinuousTransformerPredictor(encoder, args.dropout, args.output_transform_layers)
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
    trainer = UrateTrainer(model, train_data, mini_test_data, verify_data, optimizer, warmup_scheduler, main_scheduler, args.save_every, args.snapshot_path, use_devices, args, args.limit_steps)

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
    parser.add_argument("--snapshot-path", default="/data/ukbb/v2_snapshots/urate_predictor.pt")
    parser.add_argument("--model-type", default="transformer", help="\'transformer\' or \'linear\'")
    args = parser.parse_args()
    
    main(args)

def testing():
    interactive_setup()
    parser = get_default_argparser()
    parser.add_argument("--snapshot-path", default="/data/ukbb/v2_snapshots/urate_predictor.pt")
    parser.add_argument("--model-type", default="transformer", help="\'transformer\' or \'linear\'")
    args = parser.parse_args()
    args.model_type="linear" 
    args.total_epochs=50 
    args.seq_len=13290 
    args.h5_file="/data/ukbb/net_input/all_gwas.h5" 
    args.test_frac=0.3 
    args.devices="0,1"
    args.snapshot_path="tmp_script_urate_predictor_linear.pt"
    args.warmup_batches=500 
    args.warmup_min_lr=1e-10 
    args.warmup_max_lr=1e-5
    args.main_scheduler_step_size=1000 
    args.main_scheduler_gamma=0.9 
    args.report_on_batch=10
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
    trainer = UrateTrainer(model, train_data, mini_test_data, verify_data, optimizer, warmup_scheduler, main_scheduler, args.save_every, args.snapshot_path, use_devices, args, args.limit_steps)

    has_sdp = version.parse(torch.__version__) >= version.parse("2.0")
    if (args.model_type == "transformer"):
        with sdp_kernel(**backend_map[SDPBackend.FLASH_ATTENTION]) if (has_sdp and not args.no_gradscaler) else nullcontext():
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(not args.no_gradscaler)):
                trainer.train(args.total_epochs)
    else:
        trainer.train(args.total_epochs)

    import sklearn
    from sklearn.linear_model import LinearRegression
    x_tmp = train_data.dataset.full_tok_mat.numpy()
    y_tmp = train_data.dataset.phenos["urate"].to_numpy()
    use_ind = np.where(~np.isnan(y_tmp))[0]
    x = x_tmp[use_ind,:]
    y = y_tmp[use_ind]
    reg = LinearRegression().fit(x, y)

    test_x_tmp = mini_test_data.dataset.full_tok_mat.numpy()
    test_y_tmp = mini_test_data.dataset.phenos["urate"].to_numpy()
    test_use_ind = np.where(~np.isnan(test_y_tmp))[0]
    test_x = test_x_tmp[test_use_ind,:]
    test_y = test_y_tmp[test_use_ind]
    test_yhat = reg.predict(test_x)
    torch.nn.functional.mse_loss(torch.tensor(test_y), torch.tensor(test_yhat))