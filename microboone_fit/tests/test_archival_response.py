import numpy as np

from sterile_fit.archival_response import load_archival_reco_given_true
from sterile_fit.published_inputs import REPOSITORY_ROOT


def test_archival_response_is_normalized_as_reco_given_true() -> None:
    path = REPOSITORY_ROOT / "data" / "raw" / "hepdata_microboone_2022_response" / "HEPData-ins1953539-v3-nu_eCC_FC_Energy_Resolution.yaml"
    response = load_archival_reco_given_true(path)
    sums = response.reco_given_true.sum(axis=0)
    assert response.reco_given_true.shape == (60, 60)
    assert np.allclose(sums[response.valid_true_indices], 1.0)
    assert np.all(response.raw_column_sums[response.valid_true_indices] > 0.0)
    zero_true_indices = np.setdiff1d(np.arange(60), response.valid_true_indices)
    assert np.all(response.reco_given_true[:, zero_true_indices] == 0.0)
