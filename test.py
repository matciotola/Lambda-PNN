import argparse
import gc
import os
import time
from tqdm import tqdm

import numpy as np
import scipy.io as io
import torch
import torch.optim as optim

import metrics as mt

from network import LPNN
from coregistration import Coregistration
from loss import StructuralLoss
from tools.input_prepocessing import input_preparation, resize_images
from tools.spectral_tools import gen_mtf, gen_mtf_pan
from tools import utils
from tools.show_results import show
from tools.salient_patches_extraction import patches_extractor_w_kmeans


def main_l_pnn_test(args):

    # Arguments parsing
    test_path = args.input
    method = 'L-PNN'
    sensor = args.sensor
    out_dir = os.path.join(args.out_dir, sensor, method)

    gpu_number = args.gpu_number
    use_cpu = args.use_cpu
    reduce_res_flag = args.RR
    coregistration_flag = args.coregistration

    from_scratch_flag = args.from_scratch

    num_patches = args.num_patches
    patch_size = args.patch_size

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_number)

    # Hyperparameters definition
    semi_width = 8
    ratio = 4

    # Torch configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() and not use_cpu else "cpu")
    device_cpu = torch.device("cpu")

    # Parameters definition
    nbands, nbits, learning_rate, alpha, beta, gamma, epochs = utils.parameters_def(sensor,
                                                                                    args.learning_rate,
                                                                                    args.epochs
                                                                                    )

    # Network definition
    net = LPNN(nbands + 1).to(device)

    # Optimizer definition
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Loading pretrained model
    weight_path = os.path.join('weights', sensor + '_' + method + '_model.tar')
    if os.path.exists(weight_path) and not from_scratch_flag:
        net.load_state_dict(torch.load(weight_path))
    else:
        print('Training from scratch will be performed.')

    # Losses and Downgrade Protocol definitions
    coreg = Coregistration(gen_mtf_pan(ratio, sensor), device, ratio).to(device)
    downgrade = mt.DowngradeProtocol(gen_mtf(ratio, sensor), ratio, device).to(device)
    loss_d_lambda_khan = mt.ReproDLambdaKhan(device).to(device)
    loss_reprojected_ergas = mt.ERGAS(ratio).to(device)
    loss_structural = StructuralLoss(ratio).to(device)

    # Loading test data
    temp = io.loadmat(test_path)

    pan_np = temp['I_PAN'].astype('float32')
    ms_np = temp['I_MS_LR'].astype('float32')

    # Downgrade protocol for Reduced Resolution assessment
    if reduce_res_flag:
        num_patches = 4
        ms_np, pan_np = resize_images(ms_np, pan_np, ratio, sensor)

    prepro_time_start = time.time()
    # Preprocessing
    inputs, _, ms_exp, pan = input_preparation(ms_np, pan_np, ratio, nbits, device_cpu)

    # Global misalignment estimation
    if coregistration_flag:
        r, c = coreg(ms_exp.to(device), pan.to(device))
    else:
        r = torch.tensor([0])
        c = torch.tensor([0])

    r = r.repeat(num_patches, 1)
    c = c.repeat(num_patches, 1)

    # Image reshaping for operational purposes
    if (inputs.shape[-2] > args.max_dim) or (inputs.shape[-1] > args.max_dim):
        kc, kh, kw = nbands + 1, args.max_dim, args.max_dim  # kernel size
        dc, dh, dw = nbands + 1, args.max_dim, args.max_dim  # stride
        patches = inputs.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
        unfold_shape = list(patches.shape)
        patches = patches.contiguous().view(-1, kc, kh, kw).to(device)

        unfold_flag = True
    else:
        patches = torch.clone(inputs).to(device)
        unfold_flag = False
        unfold_shape = []

    # Salient patch extraction for fine-tuning
    if inputs.shape[-2] > patch_size or inputs.shape[-1] > patch_size:
        inputs = patches_extractor_w_kmeans(inputs, n_clusters=num_patches, patch_size=patch_size)
        if inputs.shape[0] > num_patches:
            inputs = inputs[:num_patches, :, :, :]

    # Generating labels and threshold mask for structural loss
    labels_spec = inputs[:, :-1, :, :]
    labels_ms = labels_spec[:, :, 2::4, 2::4]
    labels_struct = torch.unsqueeze(inputs[:, -1, :, :], dim=1)
    threshold = utils.local_corr_mask(inputs, ratio, sensor, device, semi_width)

    # Moving data to GPU
    inputs = inputs.to(device)
    del labels_spec
    labels_ms = labels_ms.to(device)
    labels_struct = labels_struct.to(device)
    threshold = threshold.to(device)

    prepro_time = time.time() - prepro_time_start

    print('Preprocessing time: {:.2f} s'.format(prepro_time))

    # Best model path implementation
    temp_path = 'weights_target_adaptive'

    if not os.path.exists(temp_path):
        os.mkdir(temp_path)
    best_model_path = os.path.join(temp_path, 'weights_' + 'target_adaptive' + '_' + method + '_' + sensor + '.tar')

    # Target Adaptation - Fine-Tuning step
    history_loss = []
    history_loss_r_ergas = []
    history_loss_lambda = []
    history_loss_struct = []

    min_loss = np.inf
    training_time_start = time.time()
    pbar = tqdm(range(epochs), dynamic_ncols=True)

    for epoch in pbar:

        optimizer.zero_grad()
        # Prediction - Forward step
        outputs = net(inputs)

        # Downsample of the outputs for the loss computation -- re-projection step
        downgraded_shifted_outputs = downgrade(outputs, r, c)

        # Loss computation
        loss_ergas = loss_reprojected_ergas(downgraded_shifted_outputs, labels_ms)
        loss_lambda = loss_d_lambda_khan(downgraded_shifted_outputs * 2048.0, labels_ms * 2048.0)
        loss_struct, loss_struct_no_threshold = loss_structural(outputs, labels_struct, threshold)
        loss = alpha * loss_ergas + beta * loss_lambda + gamma * loss_struct
        # Backward step
        loss.backward()

        # Weight update
        optimizer.step()

        # Saving best model
        if loss < min_loss:
            min_loss = loss
            torch.save(net.state_dict(), best_model_path)

        # Losses update
        running_loss = loss.item()
        running_ergas_loss = loss_ergas.item()
        running_lambda_loss = loss_lambda.item()
        running_struct_loss = loss_struct_no_threshold

        # Progress bar update
        pbar.set_description("Epoch: {:03} / {:03}".format(epoch + 1, epochs))
        pbar.set_postfix(
            {'Overall Loss': loss.item(), 'R-ERGAS': round(loss_ergas.item(), 4),
             'Khan Loss': round(loss_lambda.item(), 4), 'Structural Loss': round(loss_struct_no_threshold, 4)})

        # History update
        history_loss.append(running_loss)
        history_loss_r_ergas.append(running_ergas_loss)
        history_loss_lambda.append(running_lambda_loss)
        history_loss_struct.append(running_struct_loss)

    target_adaptation_time = time.time() - training_time_start
    print('Target Adaptation time: {:.2f} s'.format(target_adaptation_time))

    # Best model loading
    if epochs > 0 and os.path.exists(best_model_path):
        print('Loading best model')
        net.load_state_dict(torch.load(best_model_path))

    # Clearing GPU memory
    torch.cuda.empty_cache()
    # Clearing memory
    gc.collect()

    # Performing Pansharpening on the test image
    net.eval()

    # Testing
    prediction_time_start = time.time()
    if unfold_flag:
        outputs_patches = []
        with torch.no_grad():
            for i in range(patches.shape[0]):

                # Singular patch prediction
                patch = torch.unsqueeze(patches[i], 0)
                patch = patch.to(device)
                outputs = net(patch)
                outputs_patches.append(outputs)

        # Image reconstruction
        outputs_patches = torch.cat(outputs_patches, 0)
        unfold_shape[4] = unfold_shape[4] - 1
        outputs = outputs_patches.view(unfold_shape)
        output_c = unfold_shape[1] * unfold_shape[4]
        output_h = unfold_shape[2] * unfold_shape[5]
        output_w = unfold_shape[3] * unfold_shape[6]
        outputs = outputs.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
        outputs = outputs.view(1, output_c, output_h, output_w)

    else:
        # Predict the whole image
        outputs = net(patches)

    prediction_time = time.time() - prediction_time_start
    print('Prediction time: {:.2f} s'.format(prediction_time))

    # Convert to numpy array
    out = outputs.cpu().detach().numpy()
    # Reshape to image
    out = np.squeeze(out)
    out = np.moveaxis(out, 0, -1)
    # Denormalization
    out = out * (2 ** nbits)
    # Get only positive values
    out = np.clip(out, 0, out.max())

    out = out.astype(np.uint16)

    # Saving the results
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    name = test_path.split(os.sep)[-1].split('.')[0] + '_' + method + '_' + str(epochs) + '.mat'
    save_path = os.path.join(out_dir, name)

    io.savemat(save_path, {'I_MS': out})

    if epochs > 0 and args.save_loss_trend:

        io.savemat(
            out_dir + test_path.split(os.sep)[-1].split('.')[0] + '_' + method + '_' + str(
                num_patches) + '_' + str(patch_size) + '_' + str(
                epochs) + '_losses_stats' + '.mat',
            {
                'prepro_time': prepro_time,
                'training_time': target_adaptation_time,
                'predict_time': prediction_time,
                'overall_loss': history_loss,
                'r_ergas_loss': history_loss_r_ergas,
                'lambda_loss': history_loss_lambda,
                'structural_loss': history_loss_struct,
            }
        )

    if args.show_results:
        show(ms_np, pan_np, out, ratio, method)

    torch.cuda.empty_cache()
    gc.collect()

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Lambda-PNN Test code',
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
    requiredNamed = parser.add_argument_group('required named arguments')

    requiredNamed.add_argument("-i", "--input", type=str, required=True,
                               help='The path of the .mat file which contains the MS '
                                    'and PAN images. For more details, please refer '
                                    'to the GitHub documentation.')

    requiredNamed.add_argument('-s', '--sensor', type=str, required=True, choices=["WV3", "WV2", 'GE1'],
                               help='The sensor that has acquired the test image. Available sensors are '
                                    'WorldView-3 (WV3), WorldView-2 (WV2), GeoEye1 (GE1)')

    optional.add_argument("-o", "--out_dir", type=str, default='Results',
                          help='The directory in which save the outcome.')
    optional.add_argument("--epochs", type=int, default=-1, help='Number of the epochs with which perform the '
                                                                 'fine-tuning of the algorithm.')
    optional.add_argument('-n_gpu', "--gpu_number", type=int, default=0, help='Number of the GPU on which perform the '
                                                                              'algorithm.')
    optional.add_argument("--use_cpu", action="store_true",
                          help='Force the system to use CPU instead of GPU. It could solve OOM problems, but the '
                               'algorithm will be slower.')
    optional.add_argument("--RR", action="store_true", help='For evaluation only. The algorithm '
                                                            'will be performed at reduced '
                                                            'resolution.')

    optional.add_argument("--coregistration", action="store_true", help="Enable the co-registration feature.")
    optional.add_argument('--no-coregistration', dest='coregistration', action='store_false',
                          help="Disable the co-registration feature.")
    optional.set_defaults(coregistration=True)

    optional.add_argument("--save_loss_trend", action="store_true", help="Option to save the trend of losses "
                                                                         "(For Debugging Purpose).")
    optional.add_argument("--show_results", action="store_true", help="Enable the visualization of the outcomes.")
    optional.add_argument("--save_weights", action="store_true", help="Save the training weights.")
    optional.add_argument("-lr", "--learning_rate", type=float, default=-1.0,
                          help='Learning rate with which perform the training.')
    optional.add_argument("-np", "--num_patches", type=int, default=16,
                          help='Number of patches used for training.')
    optional.add_argument("-ps", "--patch_size", type=int, default=256,
                          help='Dimensions of patches used for training.')
    optional.add_argument("--max_dim", type=int, default=2048,
                          help='Maximum dimension of the input image. If the input image is larger than this value, '
                               'it will be split into squared patches of this dimension.')
    optional.add_argument("--from_scratch", action="store_true",
                          help="Train the network from scratch. Enable ReduceLROnPlateau to allow high learning-rates")

    parser._action_groups.append(optional)
    arguments = parser.parse_args()

    main_l_pnn_test(arguments)
