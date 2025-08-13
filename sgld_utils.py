import numpy as np
import torch
from utils import plot_image

def select_significant_voxels(beta_samples, gamma):
    """
    Implements Section 3.3 Bayesian FDR selection.

    Args:
      beta_samples: list of betas, the shape of each beta is (amount of units in next layer, amount of voxels in input image)
                    For example, the shape of an image is 5 * 5, and the number of units in the next layer is 3.
                    Then, the shape of beta should be 3 * 25. The length of the list is the amount of beta samples that are saved.
                    So if we saved 100 samples, the dimension of beta_samples would be (100, 3, 25)
      gamma:        float, desired false discovery rate threshold

    Returns:
      mask:  np.ndarray of bool, shape (25,), True = selected voxel
      p_hat: np.ndarray of float, shape (25,), inclusion probabilities
      delta: float, cutoff probability
      r:     int, number of voxels selected
    """
    # 1) Reformat to (100, 3, 25).
    beta_arr = np.stack(beta_samples, axis=0)
    # 2) collapse to (100, 25). if all 3 values are 0, then flag that voxel as 0, otherwise, flag as 1
    any_nz = np.any(beta_arr != 0, axis=1)
    # 3) collapse to (25, )
    # Average over 100 samples to get the probability, from 0 to 1, that each voxel is significant.
    p_hat = any_nz.astype(float).mean(axis=0)
    # 4) Sort p_hat descending
    order = np.argsort(-p_hat)  # indices that sort high→low
    p_sorted = p_hat[order]  # sorted probabilities
    # 5) Compute running FDR for top k voxels
    fdr = np.cumsum(1 - p_sorted) / np.arange(1, len(p_sorted) + 1)
    # print("fdr:", fdr)
    # 6) Find largest k with FDR(k) ≤ gamma
    valid = np.where(fdr <= gamma)[0]
    if valid.size > 0:
        r = int(valid[-1] + 1)
        delta = float(p_sorted[r - 1])
    else:
        r, delta = 0, 1.0

    # 7) Build final mask
    mask = p_hat >= delta
    return mask, p_hat, delta, r


def implement_mask(samples, gamma, model, device='cpu'):
    with torch.no_grad():
        beta_samples = samples['beta']
        mask, p_hat, delta, r = select_significant_voxels(beta_samples, gamma)
        print(f"Threshold δ={delta:.3f}, selecting r={r} voxels at FDR={gamma}")

        # 2) zero out all beta weights for voxels where mask[j] is False
        bool_mask = torch.tensor(mask, device=model.input_layer.beta.device)
        # assume beta shape is (U1, V), so we mask along the V dimension
        model.input_layer.beta.data[:, ~bool_mask] = 0
        p_masked = p_hat * mask.astype(float)
        #   now p_masked[j] == p_hat[j] if mask[j]==True, else 0.0

        # 2) reshape into your image grid
        side_length_of_image = int(np.sqrt(p_masked.size))
        prob_img_masked = p_masked.reshape(side_length_of_image, side_length_of_image)
        plot_image(prob_img_masked, "probabilities", "probabilities")
        return prob_img_masked
