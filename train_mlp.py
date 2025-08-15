from trainer import *
import os
from encoder import MLP


def get_model(args):
    mlp = MLP(args.seq_len, args.embed_dim, args.num_layers, dropout=args.dropout)
    return mlp

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
    trainer = ClassifierTrainer(model, train_data, mini_test_data, verify_data, optimizer, warmup_scheduler, main_scheduler, args.save_every, args.snapshot_path, use_devices, args, args.limit_steps)

    trainer.train(args.total_epochs)
    destroy_process_group()


def interactive_setup():
    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12357"


if __name__ == "__main__":
    parser = get_default_argparser()
    parser.add_argument("--snapshot-path", default="/data/ukbb/v2_snapshots/classifier.pt")
    args = parser.parse_args()
    
    main(args)

def testing():
    parser = get_default_argparser()
    parser.add_argument("--snapshot-path", default="/data/ukbb/v2_snapshots/classifier.pt")
    args = parser.parse_args()
    interactive_setup()
    args.devices = '4'
    ddp_setup()

    model = get_model(args)
    dataset, mini_test_set, verify_set, optimizer, warmup_scheduler, main_scheduler = load_train_objs(args, model)
