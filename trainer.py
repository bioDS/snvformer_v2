# adapted from https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/multigpu_torchrun.py
import pandas
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.backends.cuda import sdp_kernel, SDPBackend
from packaging import version
from torch.distributed import init_process_group, destroy_process_group
from torch.optim import lr_scheduler
from streaming_dataset import StreamingDataset
import os
import numpy as np
import argparse
from tqdm import tqdm
import h5py
from itertools import islice
from copy import deepcopy
from sklearn import metrics
import tempfile

# np.random.seed(42)
# torch.manual_seed(42)

rng = np.random.default_rng()

# N.B. will always mask the same indices for the whole batch
# since the batch is being trained simultaneously, this should be fine.
def mask_snvs(snvs, frac):
    length = snvs.shape[1]
    num_change = int(np.ceil(frac*length))
    change_pos = np.random.choice(length, num_change, replace=False)
    new_snvs = snvs.detach().clone()
    new_snvs[:,change_pos] = torch.zeros_like(snvs[:,0].unsqueeze(1))
    return new_snvs

def test_mask_snvs():
    snvs = torch.randint(10,(2,10))
    masked_snvs = mask_snvs(snvs, 0.5)
    assert(not snvs.equal(masked_snvs))


class FakeModule(nn.Module):
    def __init__(self, model, *args):
        super().__init__(*args)
        self.module = model

    def forward(self, *args):
        return self.module(*args)

"""
General Model Trainer
Subclasses must implement:
    - _get_source_target
    - _get_loss
"""
class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        test_data: DataLoader,
        verify_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        warmup_scheduler,
        main_scheduler,
        save_every: int,
        snapshot_path: str,
        use_devices: list,
        program_args,
        limit_steps = -1,
        gradual_length_increases = False,
        mask_frac = 0.5,
        ignore_snapshot=False,
        skip_tests=False,
        single_gpu=False,
        print_minibatches=True,
    ) -> None:
        if type(program_args) == argparse.Namespace:
            program_args = vars(program_args)
        self.print_minibatches = print_minibatches
        self.local_rank = int(os.environ["LOCAL_RANK"])
        if (use_devices != None):
            self.gpu_id = use_devices[int(os.environ["LOCAL_RANK"])]
        else:
            self.gpu_id = None
        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.test_data = test_data
        self.verify_data = verify_data
        self.optimizer = optimizer
        self.warmup_scheduler = warmup_scheduler
        self.main_scheduler = main_scheduler
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        # save losses so we get the full list even when resuming partial training.
        self.batch_losses = []
        self.tiny_test_losses = []
        self.epoch_mean_batch_losses = []
        self.epoch_mini_test_losses = []
        self.mask_frac = mask_frac
        # final test stats
        self.test_summary = None
        if os.path.exists(snapshot_path) and not ignore_snapshot:
            print(f"Loading snapshot: {snapshot_path}")
            self._load_snapshot_from_path(snapshot_path)

        if self.gpu_id != None:
            if single_gpu:
                self.model = FakeModule(model.to(self.gpu_id))
            else:
                self.model = DDP(self.model, device_ids=[self.gpu_id])
        else:
            self.model = FakeModule(model.to("cpu"))
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=(not program_args["no_gradscaler"]))
        self.limit_steps = limit_steps
        self.program_args = program_args
        self.warmup_batches = program_args["warmup_batches"]
        self.report_on_batch_step = program_args["report_on_batch"]
        self.num_loss_outputs = 1
        self.gradual_length_increases = gradual_length_increases
        self.rows_processed = 0
        # extract seq_len from train_data
        epoch_iter, (phenos, positions, chromosomes, snvs) = next(enumerate(self.train_data))
        self.seq_len = snvs.shape[1]
        self.skip_tests = skip_tests

    def _prerun(self):
        with torch.no_grad():
            self.max_len = 0
            _, (phenos, positions, chromosomes, snvs) = next(enumerate(self.train_data))
            source, targets = self._get_source_targets(phenos, positions.to(self.gpu_id), chromosomes.to(self.gpu_id), snvs.to(self.gpu_id)) # TODO: clean up max_len situation
            output = self.model(*source)
            self.model = self.model.to(self.gpu_id)

    def _load_snapshot(self, snapshot):
        self._prerun()
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        try:
            self.optimizer.load_state_dict(snapshot["OPTIMIZER_STATE"])
        except ValueError:
            print("failed to load optimizer state")
        self.warmup_scheduler.load_state_dict(snapshot["WARMUP_SCHEDULER_STATE"])
        self.main_scheduler.load_state_dict(snapshot["MAIN_SCHEDULER_STATE"])
        self.program_args = snapshot["PROGRAM_ARGS"]
        if type(self.program_args) == argparse.Namespace:
            self.program_args = vars(self.program_args)
        self.tiny_test_losses = snapshot["TINY_TEST_LOSSES"]
        self.batch_losses = snapshot["BATCH_LOSSES"]
        self.epoch_mean_batch_losses = snapshot["EPOCH_MEAN_BATCH_LOSSES"]
        self.epoch_mini_test_losses = snapshot["EPOCH_MINI_TEST_LOSSES"]
        try:
            self.test_summary = snapshot["TEST_SUMMARY"]
        except KeyError:
            self.test_summary = None
        if self.local_rank == 0:
            self.warmup_scheduler.verbose = self.program_args["verbose_scheduler"]
            self.main_scheduler.verbose = self.program_args["verbose_scheduler"]
        else:
            self.warmup_scheduler.verbose = False
            self.main_scheduler.verbose = False

    def _load_snapshot_from_path(self, snapshot_path):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        # run once to initialise any lazy layers
        self._load_snapshot(snapshot)
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    """
    Must be implemented by subclasses
    """
    def _get_source_targets(self, phenos, positions, chromosomes, snvs):
        pass

    """.
    Must be implemented by subclasses
    """
    def _get_loss(self, model_output, targets):
        pass

    def _run_batch(self, epoch_iter, source, targets, no_grad=False):
        if no_grad:
            with torch.no_grad():
                output = self.model(*source)
                total_loss, *losses = self._get_loss(output, targets)
        else:
            # print(f"ev {os.environ['LOCAL_RANK']}, dev {self.gpu_id}, running model")
            output = self.model(*source)
            # print(f"dev {self.gpu_id}, model done")
            self.optimizer.zero_grad(set_to_none=True)
            # print(f"dev {self.gpu_id}, getting loss")
            total_loss, *losses = self._get_loss(output, targets)
            # print(f"dev {self.gpu_id}, total loss: {total_loss}")
            self.grad_scaler.scale(total_loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        return total_loss, *losses

    def _count_to_length_limit(self, epoch):
        min_size = 64
        return min(min_size*(2**epoch), self.seq_len)

    """
    General method, uses _get_source_targets and _get_loss for model-specific components
    """
    def _run_epoch(self, epoch):
        world_size = int(os.environ["WORLD_SIZE"])
        b_sz = len(next(iter(self.train_data))[0])
        if (self.print_minibatches):
            print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)} | lr: {self.optimizer.param_groups[0]['lr']}")
        self.train_data.sampler.set_epoch(epoch)
        batch_total_losses = []
        individual_losses = []
        for i in range(self.num_loss_outputs):
            individual_losses.append([])
        actual_iter_steps = 0
        if self.limit_steps >= 0:
            actual_iter_len = self.limit_steps // int(os.environ["WORLD_SIZE"])
        else:
            actual_iter_len = len(self.train_data)
        if self.gradual_length_increases:
            self.max_len = self._count_to_length_limit(epoch)
        else:
            self.max_len = 0

        # print(f"pre-iter 0, dev {self.gpu_id}")
        # epoch_iter, (phenos, positions, chromosomes, snvs) = next(enumerate(self.train_data)) #TODO: remove
        for epoch_iter, (phenos, positions, chromosomes, snvs) in enumerate(self.train_data):
            seq_len = snvs.shape[1]
            # print(f"dev {self.gpu_id}, iter {epoch_iter}")
            positions = positions.to(self.gpu_id)
            chromosomes = chromosomes.to(self.gpu_id)
            snvs = snvs.to(self.gpu_id)
            source, targets = self._get_source_targets(phenos, positions, chromosomes, snvs) # TODO: clean up max_len situation
            # print(f"dev {self.gpu_id}, running batch")
            total_loss, *losses = self._run_batch(epoch_iter, source, targets)
            # print(f"dev {self.gpu_id}, total loss: {total_loss}")
            np_loss = total_loss.detach().cpu().numpy()
            if not np.isnan(np_loss):
                batch_total_losses.append(np_loss)
            for i, l in enumerate(losses):
                np_l = l.float().detach().cpu().numpy()
                if not np.isnan(np_l):
                    individual_losses[i].append(np_l)

            # N.B. we are intentionally stepping the scheduler every batch
            # because epochs are very large.
            # only use main scheduler if we're past the warmup.
            # pytorch wars we haven't called optimizer.step() first, I suspect this is because we're using gradscaler instead.
            if actual_iter_len*epoch + actual_iter_steps > self.warmup_batches: # print(f"dev {self.gpu_id}, stepping main")
                self.main_scheduler.step()
            else:
                # print(f"dev {self.gpu_id}, stepping warmup")
                self.warmup_scheduler.step()

            # if self.local_rank == 0 and epoch_iter % self.report_on_batch_step == 0:
            if epoch_iter % self.report_on_batch_step == 0 and not self.skip_tests:
                # Hyena hangs if we don't run this on all GPUs
                tiny_test_loss, *tt_losses = self._get_test_loss(limit_iter=5)
                if self.local_rank == 0:
                    if (self.print_minibatches):
                        print(f"rank {self.local_rank} - [GPU{self.gpu_id}] Epoch {epoch} | Batch {epoch_iter}/{len(self.train_data)} | lr: {self.optimizer.param_groups[0]['lr']} | max_len: {self.max_len} | Batch Loss {total_loss} | Tiny Test Loss {tiny_test_loss}")
                self.tiny_test_losses.append(tiny_test_loss)
                self.batch_losses.append(total_loss)
            #    if len(tt_losses) > 1:
                #    print([f"{li:.5f}" for li in tt_losses])

            if self.limit_steps >= 0 and epoch_iter > self.limit_steps:
                break
            actual_iter_steps += 1 # epoch_iter increases by the number of available gpus
            self.rows_processed += b_sz*world_size

        if self.local_rank == 0 and not self.skip_tests:
            mean_loss = np.mean(batch_total_losses)
            mini_test_loss, *mt_losses = self._get_test_loss()
            self.epoch_mean_batch_losses.append(mean_loss)
            self.epoch_mini_test_losses.append(mini_test_loss)
            if (self.print_minibatches):
                print(f" - [GPU{self.gpu_id}] Epoch {epoch} | Mean Loss {mean_loss} | Mini Test Loss {mini_test_loss}")
            if len(mt_losses) > 1:
                print([f"{li:.5f}" for li in mt_losses])

    """
    General method, uses _get_source_targets and _get_loss for model-specific components
    """
    def _get_test_loss(self, limit_iter=-1):
        batch_total_losses = []
        individual_losses = []
        for i in range(self.num_loss_outputs):
            individual_losses.append([])
        for epoch_iter, (phenos, positions, chromosomes, snvs) in enumerate(self.test_data):
            positions = positions.to(self.gpu_id)
            chromosomes = chromosomes.to(self.gpu_id)
            snvs = snvs.to(self.gpu_id)
            source, targets = self._get_source_targets(phenos, positions, chromosomes, snvs)
            total_loss, *losses = self._run_batch(epoch_iter, source, targets, no_grad=True)
            np_loss = total_loss.detach().cpu().numpy()
            # if not np.isnan(np_loss):
            batch_total_losses.append(np_loss)
            for i, l in enumerate(losses):
                np_l = l.float().detach().cpu().numpy()
                # if not np.isnan(np_l):
                individual_losses[i].append(np_l)
            if limit_iter > 0 and epoch_iter > limit_iter:
                break
        # return np.nanmean(batch_total_losses), *[np.nanmean(li) for li in individual_losses]
        return np.nanmean(batch_total_losses), None

    def _get_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch+1,
            "OPTIMIZER_STATE": self.optimizer.state_dict(),
            "WARMUP_SCHEDULER_STATE": self.warmup_scheduler.state_dict(),
            "MAIN_SCHEDULER_STATE": self.main_scheduler.state_dict(),
            "PROGRAM_ARGS": self.program_args,
            "TINY_TEST_LOSSES": self.tiny_test_losses,
            "BATCH_LOSSES": self.batch_losses,
            "EPOCH_MEAN_BATCH_LOSSES": self.epoch_mean_batch_losses,
            "EPOCH_MINI_TEST_LOSSES": self.epoch_mini_test_losses,
            "TEST_SUMMARY": self.test_summary
        }
        return snapshot

    def _save_snapshot_path(self, epoch, snapshot_path):
        snapshot = self._get_snapshot(epoch)
        torch.save(snapshot, snapshot_path)
        torch.save(snapshot, snapshot_path + f"_epoch-{epoch}")
        if (self.print_minibatches):
            print(f"Epoch {epoch} | Training snapshot saved at {snapshot_path}")

    def _save_snapshot(self, epoch):
        self._save_snapshot_path(epoch, self.snapshot_path)

    def get_identity_loss(self):
        epoch_iter, (phenos, positions, chromosomes, snvs) = next(enumerate(self.test_data))
        positions = positions.to(self.gpu_id)
        chromosomes = chromosomes.to(self.gpu_id)
        snvs = snvs.to(self.gpu_id)
        source, targets = self._get_source_targets(phenos, positions, chromosomes, snvs)
        if source[0].shape == targets.shape:
            identity = torch.nn.functional.one_hot(source[0].long()).float() * 100 # output is before softmax
            loss = self._get_loss(identity, targets)
            if (self.print_minibatches):
                print(f"identity loss: {loss}")

    def train(self, max_epochs: int):
        # self.get_identity_loss()
        for epoch in tqdm(range(self.epochs_run, max_epochs), disable=self.local_rank != 0):
            self._run_epoch(epoch)
            if self.local_rank == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)
        # Save some summary statistics for future use.
        if self.local_rank == 0:
            self._get_verify_stats()
            self._save_snapshot(max_epochs)

    #TODO: all_targets (and other) lists get merged with dual_output model.
    def _get_verify_stats(self):
        print("Gettting test-set stats")
        batch_total_losses = []
        individual_losses = []
        all_targets = [[],[]]
        all_preds = [[],[]]
        all_gout = []
        all_urate = []
        # run tests without dropout
        self.model.eval()
        if self.test_summary == None:
            print("test-stats not yet collected, running now")
            for i in range(self.num_loss_outputs):
                individual_losses.append([])
            # epoch_iter, (phenos, positions, chromosomes, snvs) = next(enumerate(self.verify_data))
            for epoch_iter, (phenos, positions, chromosomes, snvs) in enumerate(self.verify_data):
                positions = positions.to(self.gpu_id)
                chromosomes = chromosomes.to(self.gpu_id)
                snvs = snvs.to(self.gpu_id)
                self.max_len=0
                source, targets = self._get_source_targets(phenos, positions, chromosomes, snvs)
                gout = phenos["gout"].to_numpy(dtype=bool)
                urate = phenos["urate"].to_numpy(dtype=bool)
                with torch.no_grad():
                    output = self.model(*source)
                    total_loss, *losses = self._get_loss(output, targets)
                np_loss = total_loss.detach().cpu().numpy()
                # if not np.isnan(np_loss):
                batch_total_losses.append(np_loss)
                for i, l in enumerate(losses):
                    np_l = l.float().detach().cpu().numpy()
                    # if not np.isnan(np_l):
                    individual_losses[i].append(np_l)
                if type(targets) == tuple:
                    for i, t in enumerate(targets):
                        all_targets[i].extend(t.to("cpu"))
                    for i, t in enumerate(output):
                        all_preds[i].extend(t.to("cpu"))
                    # all_preds.extend((o.to('cpu') for o in output))
                else:
                    all_targets[0].extend(targets.to('cpu'))
                    all_preds[0].extend(output.to('cpu'))
                all_gout.extend(gout)
                all_urate.extend(urate)
            test_batch_total_losses = np.array(batch_total_losses)
            test_mean_loss = np.mean(test_batch_total_losses)
            print(f"mean test loss: {test_mean_loss}")
            self.test_summary = {
                "BATCH_TOTAL_LOSSES": batch_total_losses,
                "INDIVIDUAL_LOSSES": individual_losses,
                "ALL_TARGETS": all_targets,
                "ALL_PREDS": all_preds,
                "MEAN_LOSS": test_mean_loss,
                "ROC_AUC": None
            }
            gout_targets = None
            urate_targets = None
            gout_preds = None
            urate_preds = None
            if all_targets[0][0].dtype == torch.bool or all_targets[0][0].dtype == torch.int64:
                gout_targets = all_targets[0]
                gout_preds = all_preds[0]
            if all_targets[0][0].dtype == torch.float32:
                urate_targets = all_targets[0]
                urate_preds = all_preds[0]
            if len(all_targets[1]) > 0 and all_targets[1][0].dtype == torch.float32:
                urate_targets = all_targets[1]
                urate_preds = all_preds[1]
            if gout_preds != None:
                print('target is bool, getting roc')
                np_outputs = np.array([i.float().numpy() for i in gout_preds])
                np_targets = np.array([i.float().numpy() for i in gout_targets])
                if len(np_outputs.shape) == 2 and np_outputs.shape[1] == 2:
                    np_outputs = np_outputs[:,1]
                self.test_summary["ROC_AUC"] = metrics.roc_auc_score(np_targets, np_outputs)
                print(f"ROC AUC: {self.test_summary['ROC_AUC']}")
                fpr, tpr, _ = metrics.roc_curve(np_targets, np_outputs)
                self.test_summary["GOUT_FPR"] = fpr
                self.test_summary["GOUT_TPR"] = tpr
            # elif all_targets[0][0].dtype == torch.float32:
            if urate_preds != None:
                print("target is float, testing target and pred as ROC predictors of gout.")
                np_outputs = np.array([i.float().numpy() for i in urate_preds])
                np_targets = np.array([i.float().numpy() for i in urate_targets])
                np_gout = np.array(all_gout, dtype=bool)
                self.test_summary["URATE_ROC_AUC"] = metrics.roc_auc_score(np_gout, np_outputs)
                print(f"ROC AUC: {self.test_summary['URATE_ROC_AUC']}")
                fpr, tpr, _ = metrics.roc_curve(np_gout, np_outputs)
                self.test_summary["URATE_FPR"] = fpr
                self.test_summary["URATE_TPR"] = tpr
                self.test_summary["TARGET_ROC_AUC"] = metrics.roc_auc_score(np_gout, np_targets)
                print(f"Target ROC AUC (actual phenotype, not prediction!): {self.test_summary['TARGET_ROC_AUC']}")
        else:
            print("test stats already collected")
        self.model.train()

class EncoderTrainer(Trainer):
    def __init__(
        self,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

    def _get_source_targets(self, phenos, positions, chromosomes, snvs):
        if self.max_len > 0:
            snvs = snvs[:,:self.max_len]
            positions = positions[:,:self.max_len]
            chromosomes = chromosomes[:,:self.max_len]
        masked_snvs = mask_snvs(snvs, self.mask_frac)
        source = (phenos, positions, chromosomes, masked_snvs, self.max_len)
        targets = snvs
        return source, targets

    def _get_loss(self, model_output, targets):
        # snvs will always be at the end
        # treat every entry in the sequence like a separate item in the batch
        # otherwise total_loss gets confused
        num_snvs = targets.shape[1]
        snv_output = model_output[:,-num_snvs:,:]
        return F.cross_entropy(snv_output.reshape(-1, snv_output.shape[2]), targets.reshape(-1,).long()),

class HyenaEncoderTrainer(Trainer):
    def __init__(
        self,
        *args
    ) -> None:
        super().__init__(*args)

    def _get_source_targets(self, phenos, positions, chromosomes, snvs):
        if self.max_len > 0:
            snvs = snvs[:,:self.max_len]
            positions = positions[:,:self.max_len]
            chromosomes = chromosomes[:,:self.max_len]
        masked_snvs = mask_snvs(snvs, self.mask_frac)
        source = (masked_snvs.int(), positions)
        targets = snvs
        return source, targets

    def _get_loss(self, model_output, targets):
        # snvs will always be at the end
        # treat every entry in the sequence like a separate item in the batch
        # otherwise total_loss gets confused
        num_snvs = targets.shape[1]
        snv_output = model_output[:,-num_snvs:,:]
        return F.cross_entropy(snv_output.reshape(-1, snv_output.shape[2]), targets.reshape(-1,).long()),

class HyenaNextTokenEncoderTrainer(Trainer):
    def __init__(
        self,
        view_size,
        *args
    ) -> None:
        super().__init__(*args)
        self.view_size = view_size
        # self.step_size = step_size

    # returns the entire sequence, it will iterated through w/ a sliding window in run_batch
    def _get_source_targets(self, phenos, positions, chromosomes, snvs):
        source = (snvs.int(), positions)
        targets = None
        return source, targets

    def _step_get_source_targets(self, step_ind, max_len, source, targets):
        snvs, positions = source
        seq_len = snvs.shape[1]
        start_ind = max(0,step_ind - self.view_size)
        stop_ind = step_ind+2
        snvs = snvs[:,start_ind:stop_ind]
        positions = positions[:,start_ind:stop_ind]
        if max_len > 0:
            snvs = snvs[:,:max_len]
            positions = positions[:,:max_len]
        masked_final = snvs.clone()
        masked_final[:,-1] = 0
        source = (masked_final.int(), positions)
        targets = snvs[:,-1]
        return source, targets

    # final token cross-entropy only
    def _get_loss(self, next_predicted, step_target):
        # snvs will always be at the end
        # treat every entry in the sequence like a separate item in the batch
        # otherwise total_loss gets confused
        # next_predicted = output[:,-1,:] # not necessarily at the end
        actual_next = step_target
        return F.cross_entropy(next_predicted, actual_next.long())

    # sliding window over the sequence, predicting the next token at each point.
    def _run_batch(self, epoch_iter, source, targets, no_grad=False):
        seq_len = source[0].shape[1]
        num_steps = seq_len - 2

        max_len = self._count_to_length_limit(((actual_iter_len*epoch + actual_iter_steps)*b_sz*world_size)//4_000, seq_len) # count is the number of 4k of items processed.

        step_losses = []
        for step_ind in tqdm(range(0, num_steps), disable=self.local_rank != 0):
            step_source, step_target = self._step_get_source_targets(step_ind, max_len, source, targets)
            if no_grad:
                with torch.no_grad():
                    output = self.model(*step_source)[:,step_source[0].shape[1]-1,:] # final token of input seq only
                    total_loss = self._get_loss(output, step_target)
                    step_losses.append(total_loss)
            else:
                output = self.model(*step_source)[:,step_source[0].shape[1]-1,:] # final token of input seq only
                self.optimizer.zero_grad(set_to_none=True)
                total_loss = self._get_loss(output, step_target)
                self.grad_scaler.scale(total_loss).backward()
                self.grad_scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                step_losses.append(total_loss)
        return torch.mean(torch.tensor(step_losses)),

# TODO: use tensors for seq_view_{ind,val}, and maintain a dictionary mapping real inds to tensor inds.
class HyenaLearnedContextEncoderTrainer(Trainer):
    def __init__(
        self,
        max_view_size,
        max_stored_context_size,
        show_step_progress,
        seq_len,
        context_lr,
        context_cycle_frac,
        batches_per_cycle,
        *args
    ) -> None:
        super().__init__(*args)
        self.init_new_context_val = 0.0
        self.max_view_size = max_view_size
        self.show_step_progress = show_step_progress
        self.seq_len = seq_len
        self.context_lr = context_lr
        self.context_cycle_frac = context_cycle_frac
        self.max_stored_context_size = max_stored_context_size
        self.batches_per_cycle = batches_per_cycle
        # self.context_for_ind = torch.zeros((seq_len, max_stored_context_size))
        # initial weights for each index
        self.seq_view_dicts = np.empty(seq_len, dtype=object)
        for step_ind in range(0, seq_len):
            start_ind = max(0,step_ind - (self.max_view_size//2 - 1))
            stop_ind = min(step_ind + self.max_view_size//2 + 1, self.seq_len)
            use_inds = np.arange(start_ind, stop_ind)
            ind_dict = {i: self.init_new_context_val for i in use_inds}
            ind_dict.pop(step_ind) # we'll add the ind to predict manually to the end, don't include it here.
            for spare_ind in range(len(use_inds), self.max_stored_context_size):
                use_spare_ind = spare_ind % self.seq_len
                ind_dict[use_spare_ind] = self.init_new_context_val 
            self.seq_view_dicts[step_ind] = ind_dict


    # returns the entire sequence, it will iterated through w/ a sliding window in run_batch
    def _get_source_targets(self, phenos, positions, chromosomes, snvs):
        source = (snvs.int(), positions)
        targets = None
        return source, targets

    def _step_get_source_targets(self, step_ind, max_len, source, targets):
        snvs, positions = source
        seq_len = snvs.shape[1]
        if (max_len > 0):
            actual_view_size = min(self.max_view_size, max_len)
        else:
            actual_view_size = self.max_view_size
        # target_ind = actual_view_size//2
        target_ind = step_ind
        # sample acutal_view_size values from step_ind view_dict by weight
        # step_dict = self.seq_view_dicts[target_ind]
        # inds = np.array([i for i in step_dict.keys()])
        use_inds = self._sample_from_context(target_ind, actual_view_size)
        # weights = F.softmax(torch.tensor(([i for i in step_dict.values()])), dim=0).numpy()
        # use_inds = np.random.choice(inds, actual_view_size, replace=False, p=weights)
        use_inds = np.append(use_inds, target_ind)
        use_snvs = snvs[:,use_inds]
        use_positions = positions[:,use_inds]
        masked_final = use_snvs.clone()
        masked_final[:,-1] = 0 # we put the target at the end
        source = (masked_final.int(), use_positions)
        targets = use_snvs
        return source, targets, use_inds, target_ind

    def _update_batch_view_dicts(self):
        pass

    def _normalise(self, vec):
        return 100.0 * F.normalize(vec, dim=0)

    # final token cross-entropy only
    def _get_loss(self, output, step_target):
        # snvs will always be at the end
        # treat every entry in the sequence like a separate item in the batch
        # otherwise total_loss gets confused
        # next_predicted = output[:,-1,:] # not necessarily at the end
        return F.cross_entropy(output.permute(0,2,1), step_target.long())

    def _get_num_steps(self, seq_len):
        steps = 2 ** ((self.rows_processed // 50_000)+6)
        return min(steps, seq_len)

    def _get_max_len(self, seq_len):
        max_len = 2 ** ((self.rows_processed // 100_000)+5)
        self.max_len = max_len # for reporting elsewhere
        return min(max_len, seq_len)

    def _sample_from_context(self, seq_ind, num_samples, invert=False):
        ind_dict = self.seq_view_dicts[seq_ind]
        inds = np.array([i for i in ind_dict.keys()])
        if invert:
            weights = F.softmax(self._normalise(torch.tensor(([-i for i in ind_dict.values()]))), dim=0).numpy()
        else:
            weights = F.softmax(self._normalise(torch.tensor(([i for i in ind_dict.values()]))), dim=0).numpy()
        top_inds = np.random.choice(inds, num_samples, replace=False, p=weights)
        top_inds.sort()
        return top_inds

    def _get_new_inds_from_transitive_context(self, seq_ind, num_samples):
        # TODO: collect all two-step neighbours, sum w1 * w2 for each, add duplicates, remove things in first_dict, sample from these.
        weighted_candidates = {}
        first_dict = self.seq_view_dicts[seq_ind]
        # find valid first-step candidates (i.e. from which we can make a second step)
        valid_first_steps = []
        for k in first_dict.keys():
            if len(self.seq_view_dicts[k].keys() - first_dict.keys()) > 0:
                valid_first_steps.append(k)

        valid_first_subdict = {k: first_dict[k] for k in valid_first_steps}
        first_sm_weights = F.softmax(self._normalise(torch.tensor(([i for i in valid_first_subdict.values()]))), dim=0).numpy()

        # for (first_ind, first_weight) in first_dict.items():
        # (first_ind, first_prob) = next(iter(zip(valid_first_subdict.keys(), first_sm_weights)))
        for (first_ind, first_prob) in zip(valid_first_subdict.keys(), first_sm_weights):
            second_dict = self.seq_view_dicts[first_ind]
            use_second_keys = [i for i in second_dict.keys() - first_dict.keys()]
            if len(use_second_keys) > 0:
                use_second_weights = torch.tensor([second_dict[i] for i in use_second_keys])
                second_probs = F.softmax(self._normalise(use_second_weights), dim=0).numpy()

                for (second_ind, second_prob) in zip(use_second_keys, second_probs):
                    if second_ind in weighted_candidates:
                        weighted_candidates[second_ind] += first_prob * second_prob
                    else:
                        weighted_candidates[second_ind] = first_prob * second_prob

        if seq_ind in weighted_candidates:
            weighted_candidates.pop(seq_ind) # don't allow looking up the ind we want to predict.
        candidates = np.array([i for i in weighted_candidates.keys()])
        weights = np.array([i for i in weighted_candidates.values()])
        norm_weights = weights / sum(weights)
        choices = np.random.choice(candidates, num_samples, p=norm_weights, replace=False)
        # we'll init the choice weights at 0, since we're looking at softmax'd values anyway
        return choices


    # TODO: dict copying is probably inefficient
    def _update_context_weights(self, step_source, use_inds, target_ind):
        nonself_use_inds = use_inds[:-1]
        step_snvs, step_positions = step_source
        backbone = self.model.module.backbone
        with torch.no_grad():
            input_hidden_states = backbone.embeddings(step_snvs, position_ids=step_positions,)
            first_attn_out = backbone.layers[0].mixer(input_hidden_states)[:,:-1,:] # exclude the target ind
            snv_mean = torch.mean(first_attn_out, dim=[0,2]).cpu()
            adj_snv_mean = snv_mean - torch.mean(snv_mean, dim=0)
            # norm_adj_snv_mean = F.normalize(adj_snv_mean, dim=0, p=2.0)
            # softmax_attn = F.softmax(snv_mean, dim=0)
            step_dict = self.seq_view_dicts[target_ind]
            for (ind, mean_attn) in zip(nonself_use_inds, adj_snv_mean):
                step_dict[ind] = (1-self.context_lr) * step_dict[ind]  + self.context_lr * mean_attn
            # sorted_dict = {k: v for k,v in sorted()}
            # sorted_dict = dict(reversed(sorted(step_dict.items(), key=lambda item: item[1])))
            # self.seq_view_dicts[target_ind] = sorted_dict

    # TODO: very slow for long sequences
    def _add_transitive_context(self):
        num_change = int(np.ceil(self.max_stored_context_size * self.context_cycle_frac))
        # for ind in range(0, self.seq_len):
        for ind in range(0, 100):
            remove_inds = self._sample_from_context(ind, num_change, invert=True)
            new_inds = self._get_new_inds_from_transitive_context(ind, num_change)
            ind_dict = self.seq_view_dicts[ind]
            for rm in remove_inds:
                ind_dict.pop(rm)
            for ni in new_inds:
                ind_dict[ni] = self.init_new_context_val

    # sliding window over the sequence, predicting the next token at each point.
    def _run_batch(self, epoch_iter, source, targets, no_grad=False):
        seq_len = source[0].shape[1]
        num_steps = self._get_num_steps(seq_len)
        max_len = self._get_max_len(seq_len)

        step_losses = []
        for step_ind in tqdm(range(0, num_steps), disable=(self.show_step_progress == False or self.local_rank != 0)):
            step_source, step_target, use_inds, target_ind = self._step_get_source_targets(step_ind, max_len, source, targets)
            if no_grad:
                with torch.no_grad():
                    output = self.model(*step_source)
                    total_loss = self._get_loss(output, step_target)
                    step_losses.append(total_loss)
            else:
                output = self.model(*step_source)
                self.optimizer.zero_grad(set_to_none=True)
                total_loss = self._get_loss(output, step_target)
                self.grad_scaler.scale(total_loss).backward()
                self.grad_scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                step_losses.append(total_loss)
                self._update_context_weights(step_source, use_inds, target_ind)
        if (epoch_iter + 1) % self.batches_per_cycle == 0:
            self._add_transitive_context()
        return torch.mean(torch.tensor(step_losses)),

class HyenaMiddleTokenEncoderTrainer(Trainer):
    def __init__(
        self,
        max_view_size,
        show_step_progress,
        *args
    ) -> None:
        super().__init__(*args)
        self.max_view_size = max_view_size
        self.show_step_progress = show_step_progress
        # self.step_size = step_size

    # returns the entire sequence, it will iterated through w/ a sliding window in run_batch
    def _get_source_targets(self, phenos, positions, chromosomes, snvs):
        source = (snvs.int(), positions)
        targets = None
        return source, targets

    def _step_get_source_targets(self, step_ind, max_len, source, targets):
        snvs, positions = source
        seq_len = snvs.shape[1]
        if (max_len > 0):
            actual_view_size = min(self.max_view_size, max_len)
        else:
            actual_view_size = self.max_view_size
        start_ind = max(0,step_ind - (actual_view_size//2 - 1))
        stop_ind = step_ind + actual_view_size//2 + 1
        snvs = snvs[:,start_ind:stop_ind]
        positions = positions[:,start_ind:stop_ind]
        masked_final = snvs.clone()
        target_ind = masked_final.shape[1]//2
        masked_final[:,target_ind] = 0
        source = (masked_final.int(), positions)
        targets = snvs
        return source, targets

    # final token cross-entropy only
    def _get_loss(self, output, step_target):
        # snvs will always be at the end
        # treat every entry in the sequence like a separate item in the batch
        # otherwise total_loss gets confused
        # next_predicted = output[:,-1,:] # not necessarily at the end
        return F.cross_entropy(output.permute(0,2,1), step_target.long())

    def _get_num_steps(self, seq_len):
        steps = 2 ** ((self.rows_processed // 50_000)+6)
        return min(steps, seq_len)

    def _get_max_len(self, seq_len):
        max_len = 2 ** ((self.rows_processed // 100_000)+5)
        self.max_len = max_len # for reporting elsewhere
        return min(max_len, seq_len)

    # sliding window over the sequence, predicting the next token at each point.
    def _run_batch(self, epoch_iter, source, targets, no_grad=False):
        seq_len = source[0].shape[1]
        num_steps = self._get_num_steps(seq_len)
        max_len = self._get_max_len(seq_len)

        step_losses = []
        for step_ind in tqdm(range(0, num_steps), disable=(self.show_step_progress == False or self.local_rank != 0)):
            step_source, step_target = self._step_get_source_targets(step_ind, max_len, source, targets)
            if no_grad:
                with torch.no_grad():
                    output = self.model(*step_source)
                    total_loss = self._get_loss(output, step_target)
                    step_losses.append(total_loss)
            else:
                output = self.model(*step_source)
                self.optimizer.zero_grad(set_to_none=True)
                total_loss = self._get_loss(output, step_target)
                self.grad_scaler.scale(total_loss).backward()
                self.grad_scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                step_losses.append(total_loss)
        return torch.mean(torch.tensor(step_losses)),

class HyenaClassifierTrainer(Trainer):
    def __init__(
        self,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

    def _get_source_targets(self, phenos, positions, chromosomes, snvs):
        max_len = self.max_len
        if max_len > 0:
            snvs = snvs[:,:max_len]
            positions = positions[:,:max_len]
            targets = torch.tensor(phenos["gout"][:max_len].to_numpy(dtype=bool)).to(self.gpu_id)
        else:
            targets = torch.tensor(phenos["gout"].to_numpy(dtype=bool)).to(self.gpu_id)
        source = (snvs.int(), positions)
        return source, targets

    def _get_loss(self, model_output, targets):
        return F.cross_entropy(model_output, targets.long()),



class ClassifierTrainer(Trainer):
    def __init__(
        self,
        *args,
        target="gout",
        **kwargs,
    ) -> None:
        self.target=target
        super().__init__(*args, **kwargs)

    def _get_source_targets(self, phenos, positions, chromosomes, snvs):
        max_len = self.max_len
        source = (phenos, positions, chromosomes, snvs, max_len)
        if self.target == "gout":
            targets = torch.tensor(phenos["gout"].to_numpy(dtype=bool)).to(self.gpu_id)
        elif self.target == "sex":
            targets = torch.tensor((phenos["sex"] == "Male").to_numpy(dtype=bool)).to(self.gpu_id)
        else:
            raise Exception("taret must be one of 'gout' or 'sex'")
        return source, targets

    def _get_loss(self, model_output, targets):
        return F.binary_cross_entropy_with_logits(model_output.reshape(targets.shape), targets.float()),

class CEOnlyClassifierTrainer(Trainer):
    def __init__(
        self,
        *args,
        target="gout",
        **kwargs,
    ) -> None:
        self.target=target
        self.loss = nn.CrossEntropyLoss()
        super().__init__(*args, **kwargs)

    def _get_source_targets(self, phenos, positions, chromosomes, snvs):
        max_len = self.max_len
        source = (phenos, positions, chromosomes, snvs, max_len)
        if self.target == "gout":
            targets = torch.tensor(phenos["gout"].to_numpy(dtype=int)).to(self.gpu_id)
        elif self.target == "sex":
            targets = torch.tensor((phenos["sex"] == "Male").to_numpy(dtype=int)).to(self.gpu_id)
        else:
            raise Exception("taret must be one of 'gout' or 'sex'")
        return source, targets

    def _get_loss(self, model_output, targets):
        return self.loss(model_output, targets),

class HeightTrainer(Trainer):
    def __init__(
        self,
        *args
    ) -> None:
        super().__init__(*args)

    def _get_source_targets(self, max_len, phenos, positions, chromosomes, snvs):
        source = (phenos, positions, chromosomes, snvs, max_len)
        targets = torch.tensor(phenos["height"].to_numpy(dtype=np.float32)).to(self.gpu_id)
        return source, targets

    def _get_loss(self, model_output, targets):
        return F.mse_loss(model_output.squeeze(), targets),

class UrateTrainer(Trainer):
    def __init__(
        self,
        *args
    ) -> None:
        super().__init__(*args)

    def _get_source_targets(self, phenos, positions, chromosomes, snvs):
        max_len = self.max_len
        source = (phenos, positions, chromosomes, snvs, max_len)
        targets = torch.tensor(phenos["urate"].to_numpy(dtype=np.float32)).to(self.gpu_id)
        return source, targets

    def _get_loss(self, model_output, targets):
        return F.mse_loss(model_output.squeeze(), targets.squeeze()),


class CombinedClassifierUrateTrainer(Trainer):
    def __init__(
        self,
        *args,
        gout_weight=1.0,
        urate_weight=1.0,
    ) -> None:
        super().__init__(*args)
        self.gout_weight = gout_weight
        self.urate_weight = urate_weight
        self.num_loss_outputs = 2


    def _get_source_targets(self, phenos, positions, chromosomes, snvs):
        source = (phenos, positions, chromosomes, snvs, self.max_len)
        urate = torch.tensor(phenos["urate"].to_numpy(dtype=np.float32)).to(self.gpu_id)
        gout = torch.tensor(phenos["gout"].to_numpy(dtype=bool)).to(self.gpu_id)
        targets = (gout, urate)
        return source, targets

    def _get_loss(self, model_output, targets):
        gout, urate = targets
        pred_gout, pred_urate = model_output
        gout_loss = F.cross_entropy(pred_gout, gout.long())
        urate_loss = F.mse_loss(pred_urate.squeeze(), urate)
        return self.gout_weight * gout_loss + self.urate_weight * urate_loss, gout_loss, urate_loss


# common things training script will need

backend_map = {
    SDPBackend.MATH: {"enable_math": True, "enable_flash": False, "enable_mem_efficient": False},
    SDPBackend.FLASH_ATTENTION: {"enable_math": False, "enable_flash": True, "enable_mem_efficient": False},
    SDPBackend.EFFICIENT_ATTENTION: {
        "enable_math": False, "enable_flash": False, "enable_mem_efficient": True}
}


def ddp_setup():
    if not torch.distributed.is_initialized():
        init_process_group(backend="nccl")


def get_shuffled_inds(num_inds):
    shuffled_inds_dir = "./shuffled_inds"
    shuffled_inds_file = f"{shuffled_inds_dir}/{num_inds}.csv.gz"
    fmt = "%d"
    if os.path.exists(shuffled_inds_file):
        inds = np.loadtxt(shuffled_inds_file, dtype=np.int64)
    else:
        if not os.path.exists(shuffled_inds_dir):
            os.mkdir(shuffled_inds_dir)
        inds = np.arange(num_inds)
        rng.shuffle(inds)
        np.savetxt(shuffled_inds_file, inds, fmt=fmt)
    return inds

# N.B. results should always be the same
def get_train_test(args):
    phenos = pandas.read_hdf(args["h5_file"], key="phenos")
    gout = np.array(phenos.gout)
    gout_inds = np.where(gout)[0]
    # don't shuffle for now, indices *should* already be randomised, and we won't get matchine train/test sets this way.
    # shuffled_range = get_shuffled_inds(len(gout_inds))
    # gout_inds = gout_inds[shuffled_range]
    non_gout_inds = np.where(gout == False)[0]
    split_ind = int(np.ceil((1.0 - args["test_frac"]) * len(gout_inds)))
    train_gout_inds = gout_inds[0:split_ind]
    test_gout_inds = gout_inds[split_ind:]
    train_non_gout_inds = non_gout_inds[0:len(train_gout_inds)]
    test_non_gout_inds = non_gout_inds[len(train_gout_inds):len(train_gout_inds)+len(test_gout_inds)]
    unused_non_gout_inds = non_gout_inds[len(train_gout_inds)+len(test_gout_inds):]
    train_inds = np.concatenate((train_gout_inds, train_non_gout_inds))
    test_inds = np.concatenate((test_gout_inds, test_non_gout_inds))
    unused_inds = unused_non_gout_inds
    return train_inds, test_inds, unused_inds

# takes a sequence of [(phenos, position, chromosome, snv), (phenos, position, ...), ...]
# where each item in the sequence contains only one row
# outputs batched (phenos, position, chromosome, snv)
def collate_fn(data):
    all_phenos = []
    all_positions = []
    all_chromosomes = []
    all_snvs = []
    for batch in data:
        phenos, positions, chromosomes, snvs = batch
        all_phenos.append(phenos)
        all_positions.append(positions.unsqueeze(0))
        all_chromosomes.append(chromosomes.unsqueeze(0))
        all_snvs.append(snvs.unsqueeze(0))
    all_phenos = pandas.concat(all_phenos, axis=1).transpose()
    all_positions = torch.cat(all_positions, dim=0)
    all_chromosomes = torch.cat(all_chromosomes, dim=0)
    all_snvs = torch.cat(all_snvs, dim=0)
    all_data = (all_phenos, all_positions, all_chromosomes, all_snvs)
    return all_data


def prepare_dataloader(dataset: Dataset, batch_size: int, shuffle=True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset, shuffle=shuffle),
        collate_fn=collate_fn
    )

def load_train_objs(args, model):
    """
    load default train_set and optimizer
    """
    if type(args) == argparse.Namespace:
        args = vars(args)
    if os.environ["LOCAL_RANK"] != "0":
        verbose_scheduler = args["verbose_scheduler"]
    else:
        verbose_scheduler = False
    train_inds, test_inds, unused_inds = get_train_test(args)
    pretrain_inds = np.sort(np.concatenate((train_inds, unused_inds)))
    train_set = StreamingDataset(args["h5_file"], use_inds=pretrain_inds, augment_data=(not args["no_augment_data"]), augment_mult=args["augment_mult"])
    test_len = np.arange(len(test_inds))
    mini_test = test_len[0::6]
    verify_sub = [i for i in test_len if i not in mini_test]
    mini_test_set = StreamingDataset(args["h5_file"], use_inds=test_inds[mini_test], augment_frac=args["augment_frac"], augment_data=False)
    verify_set = StreamingDataset(args["h5_file"], use_inds=test_inds[verify_sub], augment_frac=args["augment_frac"], augment_data=False)
    if model != None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0, amsgrad=True)
        warmup_scheduler = lr_scheduler.LinearLR(optimizer, start_factor=args["warmup_min_lr"],
                                                end_factor=args["warmup_max_lr"],
                                                total_iters=args["warmup_batches"],
                                                verbose=False) # warmup scheduler prints a lot
        # note that we step every batch rather than every epoch
        main_scheduler = lr_scheduler.StepLR(optimizer, step_size=args["main_scheduler_step_size"],
                                            gamma=args["main_scheduler_gamma"],
                                            verbose=verbose_scheduler)
    else:
        optimizer = None
        warmup_scheduler = None
        main_scheduler = None
    return train_set, mini_test_set, verify_set, optimizer, warmup_scheduler, main_scheduler

def get_num_snv_classes(h5_file):
    h5_f = h5py.File(h5_file, 'r')
    num_snv_classes = h5_f['num_toks'][0]
    h5_f.close()
    return num_snv_classes

def get_default_argparser():
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--total-epochs', default=2, type=int, help='Total epochs to train the model')
    parser.add_argument('--save-every', type=int, default=1, help='How often to save a snapshot')
    parser.add_argument('--batch-size', default=32, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument("--embed-dim", type=int, default="64")
    parser.add_argument("--num-layers", type=int, default="4")
    parser.add_argument("--num-heads", type=int, default="4")
    parser.add_argument("--linformer-k", type=int, default="64")
    parser.add_argument("--seq-len", type=int, default="13290") # must be updated to match input file
    parser.add_argument("--test-frac", type=float, default="0.3")
    parser.add_argument("--h5-file", default="/data/ukbb/net_input/all_gwas.h5")
    parser.add_argument('-d', '--devices', help='comma delimited list of gpus to use', type=str, default="1,2,3,4")
    parser.add_argument("--limit-steps", type=int, default="-1")
    parser.add_argument("--encoder-snapshot", default="/data/ukbb/v2_snapshots/encoder.pt")
    parser.add_argument("--no-augment-data", action='store_true')
    parser.add_argument("--warmup-min-lr", type=float, default="1e-5")
    parser.add_argument("--warmup-max-lr", type=float, default="1e-2")
    parser.add_argument("--warmup-batches", type=int, default="500")
    parser.add_argument("--main-scheduler-step-size", type=int, default="100")
    parser.add_argument("--main-scheduler-gamma", type=float, default="0.9")
    parser.add_argument("--verbose-scheduler", action='store_true')
    parser.add_argument("--report-on-batch", type=int, default="100")
    parser.add_argument("--no-gradscaler", action='store_true', help="disables both autocast and gradscaler")
    parser.add_argument("--gradscaler", action='store_false', dest='no_gradscaler', help="disables both autocast and gradscaler")
    parser.add_argument("--dropout", type=float, default="0.1")
    parser.add_argument("--augment-frac", type=float, default="0.15", help="fraction of snv sequence to mask/change when augmenting gout data")
    parser.add_argument("--augment-mult", type=float, default="1.0", help="ratio of augmented to actual gout data (note that augmented data is a superset of actual data, so a value of 1.0 means only real data will be uesd.)")
    parser.add_argument("--encoder-type", default="classic", help="\'classic\' or \'linformer\'")
    parser.add_argument("--output-transform-layers", type=int, default="5")
    parser.add_argument("--gradual-length", action='store_true')
    parser.add_argument("--mask-frac", type=float, default="0.5")
    parser.add_argument("--position-encoding", default="embedding", help="'fixed' or 'embedding'")
    parser.add_argument("--snv-encoding", default="embedding", help="'one-hot' or 'embedding'")
    parser.add_argument("--ignore-chrom", action='store_true', help="ignore chromosome in embedding")
    parser.add_argument("--ignore-class", action='store_true', help="ignore class (cls/pheno/snv) in embedding")
    parser.add_argument("--pos-combine", default="add", help="'add' or 'cat'")
    parser.add_argument('--torch-sdp', action='store_true', dest='torch_sdp')
    parser.add_argument('--no-torch-sdp', action='store_false', dest='torch_sdp')
    parser.add_argument("--ffn-scale", type=int, default="4")
    parser.add_argument("--pos-embed-size", type=int, default=None)
    parser.add_argument("--snv-embed-size", type=int, default=None)
    parser.add_argument("--chrom-embed-size", type=int, default=None)
    parser.add_argument("--gene-embed-size", type=int, default=None)
    parser.add_argument("--gene-embed-graph", action='store_true')
    parser.add_argument('--tf-init', action='store_true', dest='tf_init')
    parser.add_argument('--no-tf-init', action='store_false', dest='tf_init')
    parser.set_defaults(torch_sdp=True)
    parser.set_defaults(tf_init=True)
    parser.set_defaults(no_gradscaler=True)
    return parser
