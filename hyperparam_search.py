from ray import tune, train
from ray.air import session
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler
# from ray.util.sgd.torch import is_distributed_trainable
# from ray.util.sgd.torch import distributed_checkpoint_dir
import tempfile

from trainer import *
import os
from encoder import *
from hyena_encoder import get_model
from packaging import version
from contextlib import nullcontext
from pathlib import Path

import vectorizer, trainer, encoder, gene_graph, gene_information
import importlib
for m in [vectorizer, trainer, encoder, gene_graph, gene_information]:
    importlib.reload(m)
from trainer import *
from encoder import *

# Hyperparameter tuning w/ Ray
class HPClassifierTrainer(ClassifierTrainer):
    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super(HPClassifierTrainer, self).__init__(*args, single_gpu=True, print_minibatches=False, **kwargs)
        checkpoint = session.get_checkpoint()
        if checkpoint:
            # checkpoint_state = checkpoint.to_dict()
            # self.load_snapshot(checkpoint_state)
            with checkpoint.as_directory() as checkpoint_dir:
                checkpoint_dir = Path(checkpoint_dir)
                self._load_snapshot_from_path(checkpoint_dir / "snapshot.pt")

    def _run_epoch(self, epoch):
        super(HPClassifierTrainer, self)._run_epoch(epoch)
        # checkpoint_data = self._get_snapshot(epoch)
        # checkpoint = Checkpoint.from_dict(checkpoint_data)
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot_path = os.path.join(tmpdir, "snapshot.pt")
            self._save_snapshot_path(epoch, snapshot_path)
            with torch.no_grad():
                test_loss, x = super(HPClassifierTrainer, self)._get_test_loss()
            session.report(
                {"loss": test_loss},
                checkpoint=Checkpoint.from_directory(tmpdir)
            )


class HPCEOnlyClassifierTrainer(CEOnlyClassifierTrainer):
    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super(HPCEOnlyClassifierTrainer, self).__init__(*args, single_gpu=True, print_minibatches=False, **kwargs)
        checkpoint = session.get_checkpoint()
        if checkpoint:
            # checkpoint_state = checkpoint.to_dict()
            # self.load_snapshot(checkpoint_state)
            with checkpoint.as_directory() as checkpoint_dir:
                checkpoint_dir = Path(checkpoint_dir)
                self._load_snapshot_from_path(checkpoint_dir / "snapshot.pt")

    def _run_epoch(self, epoch):
        super(HPCEOnlyClassifierTrainer, self)._run_epoch(epoch)
        # checkpoint_data = self._get_snapshot(epoch)
        # checkpoint = Checkpoint.from_dict(checkpoint_data)
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot_path = os.path.join(tmpdir, "snapshot.pt")
            self._save_snapshot_path(epoch, snapshot_path)
            with torch.no_grad():
                test_loss, x = super(HPCEOnlyClassifierTrainer, self)._get_test_loss()
            session.report(
                {"loss": test_loss},
                checkpoint=Checkpoint.from_directory(tmpdir)
            )

# class HPHyenaClassifierTrainer(HyenaClassifierTrainer):

def get_classifier_from_dict(config):
    if config["embed_dim"] == None:
        config["embed_dim"] = config["snv_embed_size"] + config["gene_embed_size"] 
        if config["ignore_chrom"] == False:
            config["embed_dim"] += config["chrom_embed_size"]
        if config["pos_combine"] == "cat":
            config["embed_dim"] += config["pos_embed_size"]
        if config["ignore_class"] == False:
            config["embed_dim"] += len(vectorizer.InputTypes)
        if config["embed_dim"] % config["num_heads"] != 0:
            additional_snv = config["num_heads"] - (config["embed_dim"] % config["num_heads"])
            config["snv_embed_size"] += additional_snv
            config["embed_dim"] += additional_snv
    print(f'using embedding size: {config["embed_dim"]}')
    if config["pos_embed_size"] == None:
        config["pos_embed_size"] = config["embed_dim"]
    if config["target"] == "gout":
        use_phenos = ["age", "sex", "bmi"]
    elif config["target"] == "sex":
        use_phenos = []
    else:
        raise Exception("target must be one of 'gout' or 'sex'")
    if config["model_type"] == "hyena":
        encoder = get_model(config)
    else:
        encoder = custom_encoder_from_args_dict(config, use_phenos)
    if not config["no_load_encoder"]:
        snapshot = torch.load(config["encoder_snapshot"], map_location="cpu")
        encoder.load_state_dict(snapshot["MODEL_STATE"])
    if config["model_type"] == "hyena":
        classifier = SimpleHyenaClassifier(encoder, config["embed_dim"])
    elif config["model_type"] == "flat_output":
        classifier = FlatOutputClassifier(encoder, config["embed_dim"])
    elif config["model_type"] == "dual_output":
        classifier = OldDualClassifier(encoder, config["embed_dim"])
    else:
        classifier = SimpleClassifier(encoder, config["embed_dim"])
    return classifier



def train_cifar(config, data_dir=None):
    interactive_setup()
    ddp_setup()
    config["warmup_min_lr"] = config["lr"]
    config["warmup_max_lr"] = config["lr"]
    model = get_classifier_from_dict(config)
    dataset, mini_test_set, verify_set, optimizer, warmup_scheduler, main_scheduler = load_train_objs(config, model)
    use_devices = [int(item) for item in config["devices"].split(',')]
    print(f"using devices: {use_devices}")
    if os.environ["RANK"] == "0":
        num_params = sum(p.numel() for p in model.parameters())
        print(f"model has {num_params} parameters")
    train_data = prepare_dataloader(dataset, config["batch_size"])
    mini_test_data = prepare_dataloader(mini_test_set, config["batch_size"])
    verify_data = prepare_dataloader(verify_set, config["batch_size"])
    if config["model_type"] == "hyena":
        trainer = HPHyenaClassifierTrainer(model, train_data, mini_test_data, verify_data, optimizer, warmup_scheduler, main_scheduler, config["save_every"], config["snapshot_path"], use_devices, config, config["limit_steps"])
    elif config["model_type"] == "dual_output":
        trainer = HPCEOnlyClassifierTrainer(model, train_data, mini_test_data, verify_data, optimizer, warmup_scheduler, main_scheduler, config["save_every"], config["snapshot_path"], use_devices, config, config["limit_steps"], config["gradual_length"], target=config["target"])
    else:
        trainer = HPClassifierTrainer(model, train_data, mini_test_data, verify_data, optimizer, warmup_scheduler, main_scheduler, config["save_every"], config["snapshot_path"], use_devices, config, config["limit_steps"], config["gradual_length"], target=config["target"])

    has_sdp = version.parse(torch.__version__) >= version.parse("2.0")
    with sdp_kernel(**backend_map[SDPBackend.FLASH_ATTENTION]) if (has_sdp and not config["no_gradscaler"]) else nullcontext():
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(not config["no_gradscaler"])):
            trainer.train(config["total_epochs"])
    destroy_process_group()

def sample_chrom_dim(spec, min, max):
    if spec.config.ignore_chrom == False:
        return np.random.choice([2**i for i in range(min, max)])
    else:
        return None

def sample_pos_dim(spec, min, max):
    if (spec.config.pos_combine == "cat"):
        return np.random.choice([2**i for i in range(min,max)])
    else:
        return None

def get_embed_dim(spec):
    if spec.config.pos_combine == "cat":
        return spec.config.snv_embed_size + spec.config.chrom_embed_size + spec.config.pos_embed_size + spec.config.gene_embed_size
    else:
        return spec.config.snv_embed_size + spec.config.chrom_embed_size + spec.config.gene_embed_size

def interactive_setup():
    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(np.random.randint(10021, 65535))


if __name__ == "__main__":
    parser = get_default_argparser()
    parser.add_argument("--model-type", default="custom")
    parser.add_argument("--snapshot-path", default="/data/ukbb/v2_snapshots/classifier.pt")
    parser.add_argument("--d-model", type=int, default = "128") # 128 and 256 used in HyenaDNA paper
    parser.add_argument("--pretrained-hyena", action='store_const', const=None)
    parser.add_argument("--no-load-encoder", action='store_true')
    parser.add_argument("--target", default="gout")
    args = parser.parse_args()
    # set to snvformer args by default
    #TODO: actually set to snvformer args
    args.encoder_type="linformer" 
    args.total_epochs=50
    args.num_layers=4
    args.batch_size=4
    args.embed_dim=64
    args.num_heads=4
    args.seq_len=320
    args.linformer_k=64
    args.h5_file="/data/ukbb/net_input/gwas_ldprune_320.h5" 
    args.test_frac=0.3 
    args.devices='0'
    args.encoder_snapshot="none.pt"
    args.snapshot_path="none.pt"
    args.warmup_batches=500 
    args.warmup_min_lr=1e-7 
    args.warmup_max_lr=1e-7 
    args.main_scheduler_step_size=1000 
    args.main_scheduler_gamma=1.00
    args.report_on_batch=10000
    args.no_gradscaler=True
    args.no_load_encoder=True
    args.target="gout"
    args.position_encoding="otf"
    # args.limit_steps=40
    args.pos_combine = "cat"
    args.snv_encoding = "embedding"
    args.ignore_chrom = False
    args.torch_sdp = True
    args.ffn_scale = 1
    args.dropout = 0.05
    #args.snv_embed_size = 4
    #args.chrom_embed_size = 4
    #args.pos_embed_size = 52
    #args.gene_embed_size = 4
    args.gene_embed_graph = True
    args.ignore_class = True
    args.model_type = "dual_output"
    # args.snapshot_path = "none.pt"
    args.save_every = 100
    args.tf_init = False
    config = vars(args)

    # specify ranges for some variables
    config["batch_size"] = tune.choice(2**i for i in range(0,3)) # old version ran up to 2^5
    config["num_layers"] = tune.choice(range(1,8))
    config["pos_combine"] = tune.choice(["cat", "add"])
    config["num_heads"] = tune.choice(2**i for i in range(0,4))
    # config["embed_dim"] = tune.choice(2**i for i in range(4, 9))
    config["snv_embed_size"] = tune.choice(2**i for i in range(1,5))
    config["ignore_chrom"] = tune.choice([True, False])
    config["chrom_embed_size"] = tune.sample_from(lambda spec: sample_chrom_dim(spec, 3, 6))
    # config["chrom_embed_size"] = tune.choice(2**i for i in range(1,5))
    # config["pos_embed_size"] = tune.choice(2**i for i in range(5))
    config["pos_embed_size"] = tune.sample_from(lambda spec: sample_pos_dim(spec, 3, 6))
    config["gene_embed_size"] = tune.choice([0] + [2**i for i in range(2,6)])
    config["embed_dim"] = None
    config["linformer_k"] = tune.choice(2**i for i in range(4,9))
    config["lr"] = tune.loguniform(1e-9, 1e-2)
    # config["no_gradscale"] = tune.choice([True, False]) # typo broke this, gradscaler was always disabled.
    config["position_encoding"] = tune.choice(["otf", "embedding"])
    config["dropout"] = tune.uniform(0.0, 0.2)
    config["ignore_chrom"] = tune.choice([True, False])
    config["ffn_scale"] = tune.choice([1, 2, 4, 8])
    config["gene_embed_graph"] = tune.choice([True, False])
    config["ignore_class"] = tune.choice([True, False])
    config["tf_init"] = tune.choice([True, False])
    config["model_type"] = tune.choice(["dual-output", "single-output"])
    # config["h5_file"] = tune.choice([
    #     "/data/ukbb/net_input/genotyped_p1e-1.h5", 
    #     "/data/ukbb/net_input/all_gwas.h5",
    #     "/data/ukbb/net_input/ld_full_r00001.h5"
    #     ])
    config["h5_file"] = "/data/ukbb/net_input/genotyped_p1e-1.h5"
    config["seq_len"] = 65803

    max_num_epochs = 150
    num_samples = 100
    storage_path = ("/home/kieran_elmes/gout_transformer_v2/ray_results")
    exp_name = "fault_tolerant_run"
    exp_path = os.path.join(storage_path, exp_name)

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2,
    )


    trainable_with_resources = tune.with_resources(partial(train_cifar), {"cpu": 4, "gpu": 1})
    if exp_path != 'none' and os.path.exists(exp_path) and tune.Tuner.can_restore(exp_path):
        tuner = tune.Tuner.restore(exp_path,
            trainable=trainable_with_resources,
            param_space=config,
            resume_errored=True,
            resume_unfinished=True
        )
    else:
        # tuner = tune.Tuner(
        #     partial(train_cifar),
        #     resources_per_trial={"cpu": 4, "gpu": 1},
        #     config=config,
        #     num_samples=num_samples,
        #     scheduler=scheduler,
        #     max_concurrent_trials=4,
        #     resume="AUTO",
        #     run_config=train.RunConfig(storage_path=storage_path, name=exp_name),
        # )
        tuner = tune.Tuner(
            trainable=trainable_with_resources,
            param_space=config,
            tune_config=tune.TuneConfig(
                num_samples=num_samples,
                scheduler=scheduler),
            run_config=train.RunConfig(storage_path=storage_path, name=exp_name),
        )
    tuner.fit()
    result = tuner.get_results()





    #result = tune.run(
    #    partial(train_cifar),
    #    resources_per_trial={"cpu": 4, "gpu": 1},
    #    config=config,
    #    num_samples=num_samples,
    #    scheduler=scheduler,
    #    max_concurrent_trials=4,
    #    resume="AUTO"
    #)

    # best_trial = result.get_best_trial("loss", "min", "last")
    best_trial = result.get_best_result("loss", "min")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final test-set loss: {best_trial.metrics['loss']}")

    torch.save(best_trial.config, "best_trial_config.pt")

# if __name__ == "__main__"
#     parser = get_default_argparser()
#     parser.add_argument("--model-type", default="custom")
#     parser.add_argument("--snapshot-path", default="/data/ukbb/v2_snapshots/classifier.pt")
#     parser.add_argument("--d-model", type=int, default = "128") # 128 and 256 used in HyenaDNA paper
#     parser.add_argument("--pretrained-hyena", action='store_const', const=None)
#     parser.add_argument("--no-load-encoder", action='store_true')
#     parser.add_argument("--target", default="gout")
#     args = parser.parse_args()

#     main(args)
