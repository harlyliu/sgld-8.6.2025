import numpy as np

from sgld_utils import select_significant_voxels


def test_significant_voxels():
    beta_samples = []
    for i in range(10):
        sample = np.zeros((4, 5))
        if i >= 3:
            sample[:, 1] = 1.0
            sample[:, 3] = 1.0
            sample[0, 1] = 0.0
            sample[0, 0] = 1.0
        if i >= 8:
            sample[0, 0] = 0.0
        beta_samples.append(sample)

    mask, p_hat, delta, r = select_significant_voxels(beta_samples, gamma=0.50)
    print("p_hat:", p_hat)
    print("delta:", delta)
    print("r:", r)
    print("mask:", mask)
    assert (mask.tolist() == [True, True, False, True, False])
    assert r == 3
    assert delta == 0.5
    assert p_hat.tolist() == [0.5, 0.7, 0.0, 0.7, 0.0]


if __name__ == "__main__":
    test_significant_voxels()
