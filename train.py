import argparse
import gc
import os

import numpy as np
import scipy.io as io
import torch
import torch.optim as optim

from torch.utils.data import DataLoader
from tqdm import tqdm

from network import LPNN
from loss import StructuralLoss

from tools.spectral_tools import gen_mtf
from dataset import MatDataset
import metrics as mt


def training_l_pnn(args):

    # HyperParameters definitions
    batch_sz = 4
    semi_width = 8
    number_files_training_zone = 32
    number_files_validation_zone = 8
    number_files_validation_zone_cross = 24

    basepath = args.input
    method = 'L-PNN'
    sensor = args.sensor
    out_dir = os.path.join(args.out_dir, sensor, method)
    epochs = args.epochs

    gpu_number = args.gpu_number
    use_cpu = args.use_cpu

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_number)

    # HyperParameters sensor-dependent definitions
    if sensor == 'WV3':
        nbands = 8
        ratio = 4
        nbits = 11

        learning_rate = 0.0002

        alpha = 0.05
        beta = 1.25
        gamma = 3.75

        # Zones definition
        train_zones = ['MexicoCity', 'Xian', 'Fortaleza']
        val_zones = ['MexicoCity', 'Xian', 'Fortaleza']
        ci_val_zones = ['Adelaide']

    elif sensor == 'WV2':
        nbands = 8
        ratio = 4
        nbits = 11

        learning_rate = 0.00015

        alpha = 0.05
        beta = 1.0
        gamma = 4.25

        # Zones definition
        train_zones = ['Berlin', 'London', 'Rome']
        val_zones = ['Berlin', 'London', 'Rome']
        ci_val_zones = ['Washington']
    elif sensor == 'GE1':
        nbands = 4
        ratio = 4
        nbits = 11

        learning_rate = 0.0001

        alpha = 0.075
        beta = 1.5
        gamma = 4.0

        # Zones definition
        train_zones = ['Norimberga', 'Rome', 'Waterford']
        val_zones = ['Norimberga', 'Rome', 'Waterford']
        ci_val_zones = ['Genova']
    else:
        raise ValueError('Sensor not supported')

    # HyperParameters command line dependent definitions
    if args.learning_rate != -1:
        learning_rate = args.learning_rate
    if epochs == -1:
        epochs = 500

    # Devices definition
    device = torch.device("cuda:0" if torch.cuda.is_available() and not use_cpu else "cpu")
    device_cpu = torch.device("cpu")

    # Data Management
    img_path = []
    val_path = []
    ci_val_path = []

    # Store the paths of the images for training set
    for name in train_zones:
        mat_root = os.path.join(basepath, name, 'Training', '512')
        img_names = sorted(next(os.walk(mat_root))[2])
        for i in range(number_files_training_zone):
            img_path.append(os.path.join(mat_root, img_names[i]))

    # Store the paths of the images for validation set
    for name in val_zones:
        mat_root = os.path.join(basepath, name, 'Validation', '512')
        img_names = sorted(next(os.walk(mat_root))[2])
        for i in range(number_files_validation_zone):
            val_path.append(os.path.join(mat_root, img_names[i]))

    # Store the paths of the images for cross-zone validation set
    for name in ci_val_zones:
        mat_root = os.path.join(basepath, name, 'Validation', '512')
        img_names = sorted(next(os.walk(mat_root))[2])
        for i in range(number_files_validation_zone_cross):
            ci_val_path.append(os.path.join(mat_root, img_names[i]))

    # Dataset definition
    dataset = MatDataset(img_path, sensor, device_cpu, ratio, semi_width, nbits)
    val_dataset = MatDataset(val_path, sensor, device_cpu, ratio, semi_width, nbits)
    ci_val_dataset = MatDataset(ci_val_path, sensor, device_cpu, ratio, semi_width, nbits)

    # DataLoader definition
    loader = DataLoader(dataset, batch_size=batch_sz, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=2,
                        persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True, prefetch_factor=2,
                            persistent_workers=True)
    ci_val_loader = DataLoader(ci_val_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True,
                               prefetch_factor=2, persistent_workers=True)

    # Network definition
    net = LPNN(nbands + 1).to(device)

    # Optimizer definition
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=int(0.1 * epochs),
                                                     threshold_mode='rel', cooldown=int(0.02 * epochs), min_lr=1e-7,
                                                     verbose=True)

    # Losses definition
    downgrade = mt.DowngradeProtocol(gen_mtf(ratio, sensor), ratio, device).to(device)
    loss_d_lambda_khan = mt.ReproDLambdaKhan(device).to(device)
    loss_reprojected_ergas = mt.ERGAS(ratio).to(device)
    loss_structural = StructuralLoss(ratio).to(device)

    # Best model path implementation
    custom_weights_path = 'custom_weights/'
    if not os.path.exists(custom_weights_path):
        os.mkdir(custom_weights_path)
    path_min_loss = os.path.join(custom_weights_path, 'weights_' + 'Training' + '_' + method + '_' + sensor + '.tar')

    # Misalignment estimation - Not used during the pretraining phase
    r = torch.tensor([0])
    c = torch.tensor([0])

    r = r.repeat(batch_sz, 1)
    c = c.repeat(batch_sz, 1)

    # History variables initialization
    history_loss = []
    history_loss_repro_ergas = []
    history_loss_d_lambda = []
    history_loss_struct = []

    history_val_loss = []
    history_val_loss_repro_ergas = []
    history_val_loss_d_lambda = []
    history_val_loss_struct = []

    history_ci_val_loss = []
    history_ci_val_loss_repro_ergas = []
    history_ci_val_loss_d_lambda = []
    history_ci_val_loss_struct = []

    # Auxiliary variables initialization
    min_loss = np.inf
    nbatches = len(loader)
    val_nbatches = len(val_loader)
    ci_val_nbatches = len(ci_val_loader)

    # Training loop
    for epoch in range(epochs):
        # Epoch losses initialization
        running_loss = 0.0
        running_ergas_loss = 0.0
        running_lambda_loss = 0.0
        running_struct_loss = 0.0

        running_val_loss = 0.0
        running_val_ergas_loss = 0.0
        running_val_lambda_loss = 0.0
        running_val_struct_loss = 0.0

        running_ci_val_loss = 0.0
        running_ci_val_ergas_loss = 0.0
        running_ci_val_lambda_loss = 0.0
        running_ci_val_struct_loss = 0.0

        # Training
        net.train()
        pbar = tqdm(loader, dynamic_ncols=True, initial=0)
        for inputs, threshold in pbar:

            # Move data to device
            inputs = inputs.to(device, non_blocking=True)
            threshold = threshold.to(device, non_blocking=True)

            # Separate labels
            labels_spec = inputs[:, :-1, :, :]
            labels_ms = labels_spec[:, :, 2::4, 2::4]
            labels_struct = torch.unsqueeze(inputs[:, -1, :, :], dim=1)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward step
            outputs = net(inputs)

            # Downgrade the outputs
            downgraded_shifted_outputs = downgrade(outputs, r, c)

            # Compute the losses
            loss_ergas = loss_reprojected_ergas(downgraded_shifted_outputs, labels_ms)
            loss_lambda = loss_d_lambda_khan(downgraded_shifted_outputs * 2048.0, labels_ms * 2048.0)  # Denormalize the data
            loss_struct, loss_struct_no_threshold = loss_structural(outputs, labels_struct, threshold)
            loss = alpha * loss_ergas + beta * loss_lambda + gamma * loss_struct

            # Backward step
            loss.backward()

            # Weight update
            optimizer.step()

            # Update the running losses
            running_loss += loss.item()
            running_ergas_loss += loss_ergas.item()
            running_lambda_loss += loss_lambda.item()
            running_struct_loss += loss_struct_no_threshold

            # Update the progress bar
            pbar.set_description("Epoch: {:03} / {:03}".format(epoch + 1, epochs))
            pbar.set_postfix(
                {'Overall Loss': loss.item(), 'R-ERGAS': round(loss_ergas.item(), 4),
                 'Khan Loss': round(loss_lambda.item(), 4), 'Structural Loss': round(loss_struct_no_threshold, 4)})

        # Validation
        net.eval()

        # Validation on validation set
        vbar = tqdm(val_loader, colour='green')
        vbar.set_description('Epoch: {:03} / {:03}'.format(epoch + 1, epochs))
        with torch.no_grad():
            for img, thr in vbar:
                # Move data to device
                inp = img.to(device, non_blocking=True)
                thr = thr.to(device, non_blocking=True)

                # Separate labels
                spec_ref = inp[:, :-1, :, :]
                spec_ref = spec_ref[:, :, 2::4, 2::4]
                struct_ref = torch.unsqueeze(inp[:, -1, :, :], dim=1)

                # Forward step
                out = net(inp)

                # Downgrade the outputs
                downgraded_shifted_outputs = downgrade(out, r, c)

                # Compute the losses
                val_loss_ergas = loss_reprojected_ergas(downgraded_shifted_outputs, spec_ref).item()
                val_loss_lambda = loss_d_lambda_khan(downgraded_shifted_outputs * 2048.0, spec_ref * 2048.0).item()
                _, val_loss_struct = loss_structural(out, struct_ref, thr)
                val_loss = alpha * val_loss_ergas + beta * val_loss_lambda + gamma * val_loss_struct

                # Update the validation running losses
                running_val_loss += val_loss
                running_val_ergas_loss += val_loss_ergas
                running_val_lambda_loss += val_loss_lambda
                running_val_struct_loss += val_loss_struct

        # Validation on cross-zone validation set
        ci_vbar = tqdm(ci_val_loader, colour='green')
        ci_vbar.set_description('Epoch: {:03} / {:03}'.format(epoch + 1, epochs))
        with torch.no_grad():
            for img, thr in ci_vbar:

                # Move data to device
                inp = img.to(device, non_blocking=True)
                thr = thr.to(device, non_blocking=True)

                # Separate labels
                spec_ref = inp[:, :-1, :, :]
                spec_ref = spec_ref[:, :, 2::4, 2::4]
                struct_ref = torch.unsqueeze(inp[:, -1, :, :], dim=1)

                # Forward step
                out = net(inp)

                # Downgrade the outputs
                downgraded_shifted_outputs = downgrade(out, r, c)

                # Compute the losses
                ci_val_loss_ergas = loss_reprojected_ergas(downgraded_shifted_outputs, spec_ref).item()
                ci_val_loss_lambda = loss_d_lambda_khan(downgraded_shifted_outputs * 2048.0, spec_ref * 2048.0).item()
                _, ci_val_loss_struct = loss_structural(out, struct_ref, thr)
                ci_val_loss = alpha * val_loss_ergas + beta * val_loss_lambda + gamma * val_loss_struct

                # Update the cross-zone validation running losses
                running_ci_val_loss += ci_val_loss
                running_ci_val_ergas_loss += ci_val_loss_ergas
                running_ci_val_lambda_loss += ci_val_loss_lambda
                running_ci_val_struct_loss += ci_val_loss_struct

        # Compute the average validation loss
        total_validation_loss = (running_val_loss / val_nbatches) + (running_ci_val_loss / ci_val_nbatches)

        # Best model saving
        if total_validation_loss < min_loss:
            min_loss = total_validation_loss
            torch.save(net.state_dict(), path_min_loss)

        # Update the learning rate scheduler
        scheduler.step(total_validation_loss)

        # Update the history
        history_loss.append(running_loss / nbatches)
        history_loss_repro_ergas.append(running_ergas_loss / nbatches)
        history_loss_d_lambda.append(running_lambda_loss / nbatches)
        history_loss_struct.append(running_struct_loss / nbatches)

        history_val_loss.append(val_loss / val_nbatches)
        history_val_loss_repro_ergas.append(running_val_ergas_loss / nbatches)
        history_val_loss_d_lambda.append(running_val_lambda_loss / nbatches)
        history_val_loss_struct.append(running_val_struct_loss / nbatches)

        history_ci_val_loss.append(val_loss / val_nbatches)
        history_ci_val_loss_repro_ergas.append(running_ci_val_ergas_loss / nbatches)
        history_ci_val_loss_d_lambda.append(running_ci_val_lambda_loss / nbatches)
        history_ci_val_loss_struct.append(running_ci_val_struct_loss / nbatches)

        # Print the losses
        print(
            'Epoch: {:03} / {:03}, Loss: {:.3f}, Lambda Loss: {:.3f}, ReproERGAS Loss: {:.3f}, Struct Loss: {:.3f}, '
            'Val Loss: {:.3f}, Val Loss CI: {:.3f}'.format(
                epoch + 1,
                epochs,
                history_loss[epoch],
                history_loss_d_lambda[epoch],
                history_loss_repro_ergas[epoch],
                history_loss_struct[epoch],
                history_val_loss[epoch],
                history_ci_val_loss[epoch]
            ), )

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    io.savemat(
        os.path.join(out_dir, method + '_losses_trend' + '.mat'),
        {
            'overall_loss': history_loss,
            'loss_repro_ergas': history_loss_repro_ergas,
            'loss_d_lambda': history_loss_d_lambda,
            'loss_structural': history_loss_struct,
            'val_loss_repro_ergas': history_val_loss_repro_ergas,
            'val_loss_d_lambda': history_val_loss_d_lambda,
            'val_loss_structural': history_val_loss_struct,
            'validation_loss': history_val_loss,
            'ci_val_loss_repro_ergas': history_ci_val_loss_repro_ergas,
            'ci_val_loss_d_lambda': history_ci_val_loss_d_lambda,
            'ci_val_loss_structural': history_ci_val_loss_struct,
            'ci_validation_loss': history_ci_val_loss
        }
    )

    torch.cuda.empty_cache()
    gc.collect()

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Lambda-PNN Training code',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='Lambda-PNN is an unsupervised deep learning-based pansharpening '
                                                 'method.',
                                     epilog='''\
Reference: 
Unsupervised Deep Learning-based Pansharpening with Jointly-Enhanced Spectral and Spatial Fidelity
M. Ciotola, G. Poggi, G. Scarpa 

Authors: 
Image Processing Research Group of University of Naples Federico II ('GRIP-UNINA')
University of Naples Parthenope

For further information, please contact the first author by email: matteo.ciotola[at]unina.it '''
                                     )
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required named arguments')

    required.add_argument("-i", "--input", type=str, required=True,
                          help='The path of the .mat file which contains the MS '
                                    'and PAN images. For more details, please refer '
                                    'to the GitHub documentation.')

    required.add_argument('-s', '--sensor', type=str, required=True, choices=["WV3", "WV2", 'GE1'],
                          help='The sensor that has acquired the test image. Available sensors are '
                                    'WorldView-3 (WV3), WorldView-2 (WV2), GeoEye1 (GE1)')

    optional.add_argument("-o", "--out_dir", type=str, default='Training',
                          help='The directory in which save the outcome.')

    optional.add_argument('-n_gpu', "--gpu_number", type=int, default=0, help='Number of the GPU on which perform the '
                                                                              'algorithm.')
    optional.add_argument("--use_cpu", action="store_true",
                          help='Force the system to use CPU instead of GPU. It could solve OOM problems, but the '
                               'algorithm will be slower.')

    optional.add_argument("-lr", "--learning_rate", type=float, default=-1.0,
                          help='Learning rate with which perform the training.')

    optional.add_argument("--epochs", type=int, default=-1, help='Number of the epochs with which perform the '
                                                                 'training of the algorithm.')

    parser._action_groups.append(optional)
    arguments = parser.parse_args()

    training_l_pnn(arguments)
