import argparse
from trainer import *
import os
from encoder import CustomEncoder, Classifier
from tqdm import tqdm
from packaging import version
from contextlib import nullcontext


def get_model(args):
    num_snv_classes = get_num_snv_classes(args.h5_file)
    encoder = CustomEncoder(args.embed_dim, args.num_heads,
             args.num_layers, num_snv_classes, args.seq_len,
             k=args.linformer_k, encoder_type="classic")
             
    classifier = Classifier(encoder)
    snapshot = torch.load(args.classifier_snapshot)
    classifier.load_state_dict(snapshot["MODEL_STATE"])
    return classifier


def main(args):
    use_devices = [int(item) for item in args.devices.split(',')]
    gpu_id = use_devices[0]
    model = get_model(args).to(gpu_id)
    train_inds, test_inds, unused_inds = get_train_test(args)
    test_dataset = StreamingDataset(args.h5_file, use_inds=test_inds, augment_data=False)
    print(f"using devices: {use_devices}")
    num_params = sum(p.numel() for p in model.parameters())
    print(f"model has {num_params} parameters")
    test_data = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        shuffle=False,
        collate_fn=collate_fn
    )

    batch_losses = []
    batch_accuracies = []
    has_sdp = version.parse(torch.__version__) >= version.parse("2.0")
    with sdp_kernel(**backend_map[SDPBackend.FLASH_ATTENTION]) if (has_sdp and not args.no_gradscaler) else nullcontext():
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            with torch.no_grad():
                for phenos, positions, chromosomes, snvs in tqdm(test_data):
                    predictions = model(phenos, positions.to(gpu_id), chromosomes.to(gpu_id), snvs.to(gpu_id))
                    binary_predictions = torch.argmax(predictions, dim=1)
                    gout = torch.tensor(phenos["gout"].to_numpy(dtype=np.int64)).to(gpu_id)
                    loss = torch.mean(F.cross_entropy(predictions, gout))
                    accuracy = sum(binary_predictions == gout)/len(predictions)
                    batch_losses.append(loss.cpu().numpy())
                    batch_accuracies.append(accuracy.cpu().numpy())
    mean_loss = np.mean(batch_losses)
    mean_accuracy = np.mean(batch_accuracies)
    print(f"mean loss: {mean_loss}")
    print(f"mean accuracy: {mean_accuracy}")

def test_testdataset():
    """
    checks the test set contains the right number of elements and that 50% are gout cases.
    """
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--batch-size', default=32, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument("--test-frac", type=float, default="0.3")
    parser.add_argument("--h5-file", default="/data/ukbb/net_input/all_gwas.h5")
    args = parser.parse_args("")
    train_inds, test_inds, unused_inds = get_train_test(args)
    test_dataset = StreamingDataset(args.h5_file, use_inds=test_inds, augment_data=False)
    test_data = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        shuffle=False,
        collate_fn=collate_fn
    )
    sum_gout = 0
    sum_nongout = 0
    for ind, (phenos, positions, chromosomes, snvs) in enumerate(test_data):
        sum_gout += np.sum(phenos["gout"] == True)
        sum_nongout += np.sum(phenos["gout"] == False)
    assert(sum_gout + sum_nongout == 5632)
    assert(sum_gout == sum_nongout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--batch-size', default=32, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument("--embed-dim", type=int, default="256")
    parser.add_argument("--num-layers", type=int, default="2")
    parser.add_argument("--num-heads", type=int, default="16")
    parser.add_argument("--linformer-k", type=int, default="64")
    parser.add_argument("--seq-len", type=int, default="13290") # must be updated to match input file
    parser.add_argument("--num-snv-classes", type=int, default="3") # choose correct number for encoding. v5 = 3.
    parser.add_argument("--test-frac", type=float, default="0.3")
    parser.add_argument("--h5-file", default="/data/ukbb/net_input/all_gwas.h5")
    parser.add_argument("--snapshot-path", default="/data/ukbb/v2_snapshots/classifier.pt")
    parser.add_argument('-d', '--devices', help='comma delimited list of gpus to use', type=str, default="1,2,3,4")
    parser.add_argument("--classifier-snapshot", default="/data/ukbb/v2_snapshots/classifier.pt")
    args = parser.parse_args()
    
    main(args)