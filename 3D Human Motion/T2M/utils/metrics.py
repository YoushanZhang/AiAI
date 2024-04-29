import numpy as np
from scipy import linalg


# (X - X_train)*(X - X_train) = -2X*X_train + X*X + X_train*X_train
def euclidean_distance_matrix(matrix1, matrix2):
    """
        Params:
        -- matrix1: N1 x D
        -- matrix2: N2 x D
        Returns:
        -- dist: N1 x N2
        dist[i, j] == distance(matrix1[i], matrix2[j])
    """
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * np.dot(matrix1, matrix2.T)    # shape (num_test, num_train)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)    # shape (num_test, 1)
    d3 = np.sum(np.square(matrix2), axis=1)     # shape (num_train, )
    dists = np.sqrt(d1 + d2 + d3)  # broadcasting
    return dists

def calculate_top_k(mat, top_k):
    size = mat.shape[0]
    gt_mat = np.expand_dims(np.arange(size), 1).repeat(size, 1)
    bool_mat = (mat == gt_mat)
    correct_vec = False
    top_k_list = []
    for i in range(top_k):
#         print(correct_vec, bool_mat[:, i])
        correct_vec = (correct_vec | bool_mat[:, i])
        # print(correct_vec)
        top_k_list.append(correct_vec[:, None])
    top_k_mat = np.concatenate(top_k_list, axis=1)
    return top_k_mat


def calculate_R_precision(embedding1, embedding2, top_k, sum_all=False):
    dist_mat = euclidean_distance_matrix(embedding1, embedding2)
    argmax = np.argsort(dist_mat, axis=1)
    top_k_mat = calculate_top_k(argmax, top_k)
    if sum_all:
        return top_k_mat.sum(axis=0)
    else:
        return top_k_mat


def calculate_matching_score(embedding1, embedding2, sum_all=False):
    assert len(embedding1.shape) == 2
    assert embedding1.shape[0] == embedding2.shape[0]
    assert embedding1.shape[1] == embedding2.shape[1]

    dist = linalg.norm(embedding1 - embedding2, axis=1)
    if sum_all:
        return dist.sum(axis=0)
    else:
        return dist

#added
# def calculate_activation_statistics(activations):
#     """
#     Params:
#     -- activations: num_samples x dim_feat
#     Returns:
#     -- mu: dim_feat
#     -- sigma: dim_feat x dim_feat
#     """
#     if np.isnan(activations).any():
#         print("Warning: NaN values detected in activation data. Replacing with zeros.")
#         activations = np.nan_to_num(activations)  # Replace NaNs with 0 and infs with large finite numbers

#     mu = np.mean(activations, axis=0)
#     if np.size(activations, axis=0) > 1:
#         cov = np.cov(activations, rowvar=False)
#     else:
#         cov = np.zeros((activations.shape[1], activations.shape[1]), dtype=activations.dtype)
    
#     return mu, cov

#added
def calculate_activation_statistics(activations):
    """
    Calculate mean and covariance of activations with handling for different dimensions.
    """
    if activations.ndim > 2:
        # Flatten or pool activations if they are not in the shape (num_samples, features)
        activations = activations.reshape(activations.shape[0], -1)

    if np.isnan(activations).any():
        print("Warning: NaN values detected in activation data. Replacing with zeros.")
        activations = np.nan_to_num(activations)

    mu = np.mean(activations, axis=0)
    if np.size(activations, axis=0) > 1:
        cov = np.cov(activations, rowvar=False)
    else:
        cov = np.zeros((activations.shape[1], activations.shape[1]), dtype=activations.dtype)

    return mu, cov



# def calculate_activation_statistics(activations):
#     """
#     Params:
#     -- activation: num_samples x dim_feat
#     Returns:
#     -- mu: dim_feat
#     -- sigma: dim_feat x dim_feat
#     """
#     # mu = np.mean(activations, axis=0)
#     # cov = np.cov(activations, rowvar=False)
#     # return mu, cov
    
#     if np.isnan(activations).any():
#         print("Warning: NaN values detected in activation data.")
#         return np.nan, np.nan  # Returning NaN if the data is not valid
#     mu = np.mean(activations, axis=0)
#     if np.size(activations, axis=0) > 1:
#         cov = np.cov(activations, rowvar=False)
#     else:
#         cov = np.zeros((activations.shape[1], activations.shape[1]))
#     return mu, cov    


def calculate_diversity(activation, diversity_times):
    assert len(activation.shape) == 2, "Activation must be 2-dimensional"
    assert activation.shape[0] > diversity_times, "Not enough samples for the requested diversity times"
    
    # Clean the activation data to replace NaNs and Infs
    activation = np.nan_to_num(activation, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Debugging print to confirm no NaNs or Infs
    if np.isnan(activation).any() or np.isinf(activation).any():
        print("Error: NaNs or Infs present even after sanitization.")
        return None

    first_indices = np.random.choice(activation.shape[0], diversity_times, replace=False)
    second_indices = np.random.choice(activation.shape[0], diversity_times, replace=False)
    
    distances = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
    return distances.mean()



def calculate_multimodality(activation, multimodality_times):
    assert len(activation.shape) == 3
    assert activation.shape[1] > multimodality_times
    num_per_sent = activation.shape[1]

    first_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    second_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    dist = linalg.norm(activation[:, first_dices] - activation[:, second_dices], axis=2)
    return dist.mean()




#added 

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance."""
    # Replace NaNs and handle Infs
    mu1 = np.nan_to_num(mu1)
    mu2 = np.nan_to_num(mu2)
    sigma1 = np.nan_to_num(sigma1)
    sigma2 = np.nan_to_num(sigma2)

    # Ensure minimum dimensionality
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    # Debug shapes
    print("Shape of mu1:", mu1.shape)
    print("Shape of mu2:", mu2.shape)
    print("Shape of sigma1:", sigma1.shape)
    print("Shape of sigma2:", sigma2.shape)

    # Check dimensions
    assert mu1.shape == mu2.shape, 'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, 'Training and test covariances have different dimensions'

    diff = mu1 - mu2    
    
    # Compute square root of product of covariances
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        print('FID calculation produces singular product; retrying with added epsilon on the diagonal.')
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Handle complex numbers
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component detected: {}'.format(m))
        covmean = covmean.real

    # Calculate Frechet Distance
    tr_covmean = np.trace(covmean)
    fid_score = (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)
    return fid_score


# def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
#     """Numpy implementation of the Frechet Distance.
#     The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
#     and X_2 ~ N(mu_2, C_2) is
#             d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
#     Stable version by Dougal J. Sutherland.
#     Params:
#     -- mu1   : Numpy array containing the activations of a layer of the
#               inception net (like returned by the function 'get_predictions')
#               for generated samples.
#     -- mu2   : The sample mean over activations, precalculated on an
#               representative data set.
#     -- sigma1: The covariance matrix over activations for generated samples.
#     -- sigma2: The covariance matrix over activations, precalculated on an
#               representative data set.
#     Returns:
#     --   : The Frechet Distance.
#     """

#     mu1 = np.atleast_1d(mu1)
#     mu2 = np.atleast_1d(mu2)
#     sigma1 = np.atleast_2d(sigma1)
#     sigma2 = np.atleast_2d(sigma2)

#     print(f"Input shapes - mu1: {mu1.shape}, mu2: {mu2.shape}, sigma1: {sigma1.shape}, sigma2: {sigma2.shape}")
#     print(f"Initial sigma1 (sample): {sigma1[:1, :1]}")
#     print(f"Initial sigma2 (sample): {sigma2[:1, :1]}")

#     assert mu1.shape == mu2.shape, 'Training and test mean vectors have different lengths'
#     assert sigma1.shape == sigma2.shape, 'Training and test covariances have different dimensions'

#     diff = mu1 - mu2    
    
#     covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
#     if not np.isfinite(covmean).all():
#         print('FID calculation produces singular product; adding epsilon to diagonal of cov estimates')
#         offset = np.eye(sigma1.shape[0]) * eps
#         covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

#     if np.iscomplexobj(covmean):
#         if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
#             m = np.max(np.abs(covmean.imag))
#             raise ValueError('Imaginary component {}'.format(m))
#         covmean = covmean.real

#     tr_covmean = np.trace(covmean)
#     fid_score = (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)
#     print(f'Computed FID score: {fid_score}')
#     return fid_score
    
    
    
#     # added
#     # print("Initial shapes: mu1 {}, mu2 {}, sigma1 {}, sigma2 {}".format(mu1.shape, mu2.shape, sigma1.shape, sigma2.shape))
#     # sigma1_nan = np.isnan(sigma1).sum()
#     # sigma2_nan = np.isnan(sigma2).sum()
#     # print("Count of NaN values: Sigma1 {}, Sigma2 {}".format(sigma1_nan, sigma2_nan))
#     # sigma1 = np.nan_to_num(sigma1, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)
#     # sigma2 = np.nan_to_num(sigma2, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)


    
    # added
    # sigma1_nan = np.isnan(sigma1).sum()
    # print ("Count of NaN values with Sigma1:",sigma1_nan)
    # sigma2_nan = np.isnan(sigma2).sum()
    # print ("Count of NaN values with Sigma2:",sigma2_nan)
    
    # sigma1_count = np.size(sigma1)
    # print ("Count of values with Sigma1:",sigma1_count)
    # sigma2_count = np.size(sigma2)
    # print ("Count of values with Sigma2:",sigma2_count)
    
    # sigma1_isinf = np.isinf(sigma1).sum()
    # print ("Count of Inf values with Sigma1:",sigma1_isinf)
    # sigma2_isinf = np.isinf(sigma2).sum()
    # print ("Count of Inf values with Sigma2:",sigma2_isinf)
    
    # sigma1 = np.nan_to_num(sigma1, nan=0.0)
    # sigma2 = np.nan_to_num(sigma2, nan=0.0)

    # sigma1_count = np.size(sigma1)
    # print ("Count of values with Sigma1 after removing Sigma 1:",sigma1_count)
    # sigma2_count = np.size(sigma2)
    # print ("Count of values with Sigma2 after removing Sigma 2:",sigma2_count)
    
