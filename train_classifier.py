from trainer import *
import os
from encoder import *
from hyena_encoder import get_model
from packaging import version
from contextlib import nullcontext

def prerun_get_classifier(encoder, ClassifierType, TrainerType, args):
    if type(args) == argparse.Namespace:
        args = vars(args)
    use_devices = [int(item) for item in args['devices'].split(',')]
    model = ClassifierType(encoder, args['embed_dim'])
    prerun_set = StreamingDataset(args['h5_file'], use_inds=[0,1], augment_data=(not args['no_augment_data']), augment_mult=args['augment_mult'])
    prerun_data = prepare_dataloader(prerun_set, args['batch_size'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0, amsgrad=True)
    verbose_scheduler = False
    warmup_scheduler = lr_scheduler.LinearLR(optimizer, start_factor=args['warmup_min_lr'],
                                            end_factor=args['warmup_max_lr'],
                                            total_iters=args['warmup_batches'],
                                            verbose=False) # warmup scheduler prints a lot
    # note that we step every batch rather than every epoch
    main_scheduler = lr_scheduler.StepLR(optimizer, step_size=args['main_scheduler_step_size'],
                                        gamma=args['main_scheduler_gamma'],
                                        verbose=verbose_scheduler)
    tmp_trainer = TrainerType(model, prerun_data, prerun_data, prerun_data, optimizer, warmup_scheduler, main_scheduler, args['save_every'], args['snapshot_path'], use_devices, args, args['limit_steps'], ignore_snapshot=True)
    tmp_trainer._prerun()
    return tmp_trainer.model.module

def get_classifier(args):
    if args.target == "gout":
        use_phenos = ["age", "sex", "bmi"]
    elif args.target == "sex":
        use_phenos = []
    else:
        raise Exception("target must be one of 'gout' or 'sex'")
    if args.model_type == "hyena":
        encoder = get_model(args)
    else:
        encoder = custom_encoder_from_args(args, use_phenos)
    if args.model_type == "hyena":
        classifier = prerun_get_classifier(encoder, SimpleHyenaClassifier, HyenaClassifierTrainer, args)
    elif args.model_type == "flat_output":
        classifier = prerun_get_classifier(encoder, FlatOutputClassifier, ClassifierTrainer, args)
    elif args.model_type == "dual_output":
        classifier = prerun_get_classifier(encoder, OldDualClassifier, CEOnlyClassifierTrainer, args)
    else:
        classifier = prerun_get_classifier(encoder, SimpleClassifier, ClassifierTrainer, args)
    # load classifier _after_ a trainer has initialised the lazy weights
    if not args.no_load_encoder:
        snapshot = torch.load(args.encoder_snapshot, map_location="cpu")
        classifier.encoder.load_state_dict(snapshot["MODEL_STATE"])
        print("encoder snapshot loaded")
    return classifier


def main(args):
    ddp_setup()
    model = get_classifier(args)
    dataset, mini_test_set, verify_set, optimizer, warmup_scheduler, main_scheduler = load_train_objs(args, model)
    use_devices = [int(item) for item in args.devices.split(',')]
    print(f"using devices: {use_devices}")
    if os.environ["RANK"] == "0":
        num_params = sum(p.numel() for p in model.parameters())
        print(f"model has {num_params} parameters")
    train_data = prepare_dataloader(dataset, args.batch_size)
    mini_test_data = prepare_dataloader(mini_test_set, args.batch_size)
    verify_data = prepare_dataloader(verify_set, args.batch_size)
    if args.model_type == "hyena":
        trainer = HyenaClassifierTrainer(model, train_data, mini_test_data, verify_data, optimizer, warmup_scheduler, main_scheduler, args.save_every, args.snapshot_path, use_devices, args, args.limit_steps)
    elif args.model_type == "dual_output":
        trainer = CEOnlyClassifierTrainer(model, train_data, mini_test_data, verify_data, optimizer, warmup_scheduler, main_scheduler, args.save_every, args.snapshot_path, use_devices, args, args.limit_steps, args.gradual_length, target=args.target)
    else:
        trainer = ClassifierTrainer(model, train_data, mini_test_data, verify_data, optimizer, warmup_scheduler, main_scheduler, args.save_every, args.snapshot_path, use_devices, args, args.limit_steps, args.gradual_length, target=args.target)
    # load encoder here, trainer needed initialised first

    has_sdp = version.parse(torch.__version__) >= version.parse("2.0")
    with sdp_kernel(**backend_map[SDPBackend.FLASH_ATTENTION]) if (has_sdp and not args.no_gradscaler) else nullcontext():
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(not args.no_gradscaler)):
            trainer.train(args.total_epochs)
    destroy_process_group()

