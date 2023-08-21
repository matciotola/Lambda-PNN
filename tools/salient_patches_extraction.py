import warnings

import torchvision.models as md
import torch
from tools.torch_kmeans.kmeans import kmeans
from tools.torch_kmeans.tools import regular_chessboard_initialization
from tools.utils import ImageClassification

warnings.simplefilter("ignore", UserWarning)


def patches_extractor_w_kmeans(i_in, n_clusters=16, patch_size=256,
                               kmeans_centers_fn=regular_chessboard_initialization):
    """
    Extracts patches from an input image using K-means clustering on features obtained from a pre-trained MobileNet
    V3 model.

        Args:
            i_in (torch.Tensor): Input image tensor of shape (batch_size, channels, height, width).
            n_clusters (int, optional): Number of clusters for K-means. Default is 16.
            patch_size (int, optional): Size of the patches to be extracted. Default is 256.
            kmeans_centers_fn (function, optional): Function to initialize K-means cluster centers.
                Default is regular_chessboard_initialization.

        Returns:
            torch.Tensor: Extracted patches of shape (n_clusters, channels, patch_size, patch_size).

        Note: - The MobileNet V3 model used here is pre-trained on the ImageNet dataset. - The input image is split
        into patches, which are then passed through the pre-trained MobileNet V3 model to obtain feature
        representations. - The features are further reduced using Principal Component Analysis (PCA) to 3 dimensions
        for clustering. - K-means clustering is performed on the reduced feature space to group similar patches into
        clusters. - For each cluster, the patch that is closest to the cluster center in the reduced feature space is
        selected. - The selected patches from each cluster are returned as the output.
    """
    transforms = ImageClassification(crop_size=224)

    device = i_in.device
    if i_in[:, :-1, :, :].shape[-1] == 8:
        rgb = (4, 2, 1)
    else:
        rgb = (2, 1, 0)

    inp_exp = torch.cat(torch.split(i_in, patch_size, -1), 0)
    inp_exp = torch.cat(torch.split(inp_exp, patch_size, -2), 0)
    inp = transforms(inp_exp[:, rgb, :, :])
    model = md.mobilenet_v3_small("MobileNet_V3_Small_Weights.IMAGENET1K_V1")
    model = model.eval().to(device)
    with torch.no_grad():
        feat = torch.squeeze(model.avgpool(model.features(inp)))
    _, s, v = torch.pca_lowrank(feat, niter=50)
    feat_pca = torch.matmul(feat, v[:, :3])
    kmeans_centers = kmeans_centers_fn(feat_pca, n_clusters)

    _, cluster_centers = kmeans(X=feat_pca, num_clusters=n_clusters, distance='euclidean', tol=1e-9,
                                            cluster_centers=kmeans_centers, device=i_in.device, tqdm_flag=False)

    cluster_centers = cluster_centers.to(device)

    indexes = []
    for i in range(cluster_centers.shape[0]):
        _, min_index = torch.min(torch.sqrt(torch.sum((feat_pca - cluster_centers[i]) ** 2, dim=1, keepdim=True)),dim=0)
        indexes.append(min_index)
        feat_pca[min_index, :] = torch.inf

    patches = inp_exp[indexes, :, :, :]

    return patches
