"""Plant definitions."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    FloatArray = NDArray[np.float64]
from typing import Mapping
import numpy as np

COMPONENTS: tuple[str, ...] = (
    "S_O",
    "S_F",
    "S_A",
    "S_NH4",
    "S_NO2",
    "S_NO3",
    "S_N2",
    "S_PO4",
    "S_I",
    "S_ALK",
    "X_I",
    "X_S",
    "X_H",
    "X_PAO",
    "X_PP",
    "X_PHA",
    "X_AOB",
    "X_NOB",
    "X_MeP",
    "X_MeOH",
)
COMPONENT_INDEX: Mapping[str, int] = {name: i for i, name in enumerate(COMPONENTS)}
N_COMPONENTS = len(COMPONENTS)
N_PROCESSES = 28
N_STAGES = 5
N_LAYERS = 10
STATE_SIZE = N_STAGES * N_COMPONENTS + N_LAYERS
TARGET_SIZE = N_COMPONENTS * (1 + N_STAGES + 2) + N_LAYERS
SOLUBLE = np.arange(10, dtype=int)
PARTICULATE = np.arange(10, 20, dtype=int)
INFLUENT_LOWER = np.asarray(
    [
        0.0,
        20.0,
        5.0,
        12.0,
        0.0,
        0.0,
        0.0,
        2.0,
        10.0,
        1.6,
        20.0,
        60.0,
        15.0,
        5.0,
        2.0,
        1.0,
        0.5,
        0.5,
        0.0,
        0.0,
    ],
    dtype=float,
)
INFLUENT_UPPER = np.asarray(
    [
        0.5,
        180.0,
        80.0,
        55.0,
        3.0,
        8.0,
        2.0,
        18.0,
        90.0,
        5.2,
        120.0,
        280.0,
        100.0,
        60.0,
        20.0,
        30.0,
        8.0,
        8.0,
        12.0,
        12.0,
    ],
    dtype=float,
)
NOMINAL_INFLUENT = (INFLUENT_LOWER + INFLUENT_UPPER) / 2.0
I_PMEP = 1.0 / 4.87
COMPOSITE_MATRIX = np.asarray(
    [
        [0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0.01, 0, 0.02, 0, 0.07, 0.07, 0, 0, 0.07, 0.07, 0, 0],
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0.01,
            0,
            0.02,
            0.02,
            1,
            0,
            0.02,
            0.02,
            I_PMEP,
            0,
        ],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.75, 0.75, 0.9, 0.9, 3.23, 0.6, 0.9, 0.9, 1, 1],
    ],
    dtype=float,
)
TSS_VECTOR = COMPOSITE_MATRIX[3].copy()
PARAMETERS: Mapping[str, float] = {
    "Y_H": 0.625,
    "Y_PAO": 0.625,
    "Y_PHA": 0.2,
    "Y_PO4": 0.4,
    "Y_AOB": 0.18,
    "Y_NOB": 0.08,
    "f_SI": 0.0,
    "f_XI": 0.1,
    "i_NSI": 0.01,
    "i_NSF": 0.0,
    "i_NXI": 0.02,
    "i_NXS": 0.0,
    "i_NBM": 0.07,
    "i_PSI": 0.0,
    "i_PSF": 0.0,
    "i_PXI": 0.01,
    "i_PXS": 0.0,
    "i_PBM": 0.02,
    "i_PMeP": I_PMEP,
    "K_H": 3.0,
    "eta_hyd_NO3": 0.6,
    "eta_hyd_NO2": 0.6,
    "eta_hyd_fe": 0.4,
    "K_O_hyd": 0.2,
    "K_NO3_hyd": 0.5,
    "K_NO2_hyd": 0.5,
    "K_NOx_hyd": 0.5,
    "K_X": 0.1,
    "mu_H": 6.0,
    "q_fe": 3.0,
    "b_H": 0.4,
    "eta_H_NO3": 0.9,
    "eta_H_NO2": 0.9,
    "K_O_H": 0.2,
    "K_F": 4.0,
    "K_fe": 4.0,
    "K_A": 4.0,
    "K_NO3_H": 0.5,
    "K_NO2_H": 0.5,
    "K_NOx_H": 0.5,
    "K_NH4_H": 0.05,
    "K_PO4_H": 0.01,
    "K_ALK_H": 0.1,
    "q_PHA": 5.0,
    "q_PP": 0.6,
    "mu_PAO": 0.56,
    "eta_PAO_NO3": 0.07,
    "eta_PAO_NO2": 0.9,
    "b_PAO": 0.2,
    "b_PP": 0.2,
    "b_PHA": 0.2,
    "K_O_PAO": 0.2,
    "K_NO3_PAO": 0.5,
    "K_NO2_PAO": 0.5,
    "K_NOx_PAO": 0.5,
    "K_NH4_PAO": 0.05,
    "K_PS": 0.2,
    "K_PO4_PAO": 0.01,
    "K_ALK_PAO": 0.1,
    "K_PP": 0.01,
    "K_max": 0.34,
    "K_IPP": 0.02,
    "K_PHA": 0.01,
    "mu_AOB": 1.81,
    "mu_NOB": 1.52,
    "b_AOB": 0.2,
    "b_NOB": 0.17,
    "K_O_AOB": 0.74,
    "K_O_NOB": 1.75,
    "K_NH4_AOB": 0.5,
    "K_NO2_NOB": 0.5,
    "K_NH4_NOB": 0.05,
    "K_ALK_nit": 0.5,
    "K_PO4_nit": 0.01,
    "k_PRE": 1.0,
    "k_RED": 0.6,
    "gamma_OH": 3.45,
    "K_ALK_PRE": 0.5,
    "K_ALK_chem": 0.5,
}


def _e(name: str) -> FloatArray:
    vector = np.zeros(N_COMPONENTS, dtype=float)
    vector[COMPONENT_INDEX[name]] = 1.0
    return vector


def build_stoichiometric_matrix(
    parameters: Mapping[str, float] = PARAMETERS,
) -> FloatArray:
    """Assemble the complete 28-by-20 Petersen matrix from continuity rules."""
    p = parameters
    nu = np.zeros((N_PROCESSES, N_COMPONENTS), dtype=float)
    e = {name: _e(name) for name in COMPONENTS}
    yh, ypao, ypha = (p["Y_H"], p["Y_PAO"], p["Y_PHA"])
    beta_h = (1.0 - yh) / (8.0 / 7.0 * yh)
    gamma_h = (1.0 - yh) / (1.72 * yh)
    beta_s, gamma_s = (ypha / (8.0 / 7.0), ypha / 1.72)
    beta_p = (1.0 - ypao) / (8.0 / 7.0 * ypao)
    gamma_p = (1.0 - ypao) / (1.72 * ypao)
    hydrolysis = (1.0 - p["f_SI"]) * e["S_F"] + p["f_SI"] * e["S_I"] - e["X_S"]
    nu[0:4] = hydrolysis
    nu[4] = -(1.0 / yh) * e["S_F"] + (1.0 - 1.0 / yh) * e["S_O"] + e["X_H"]
    nu[5] = -(1.0 / yh) * e["S_A"] + (1.0 - 1.0 / yh) * e["S_O"] + e["X_H"]
    nu[6] = -(1.0 / yh) * e["S_F"] + beta_h * (e["S_NO2"] - e["S_NO3"]) + e["X_H"]
    nu[7] = -(1.0 / yh) * e["S_F"] + gamma_h * (e["S_N2"] - e["S_NO2"]) + e["X_H"]
    nu[8] = -(1.0 / yh) * e["S_A"] + beta_h * (e["S_NO2"] - e["S_NO3"]) + e["X_H"]
    nu[9] = -(1.0 / yh) * e["S_A"] + gamma_h * (e["S_N2"] - e["S_NO2"]) + e["X_H"]
    nu[10] = e["S_A"] - e["S_F"]
    decay_h = p["f_XI"] * e["X_I"] + (1.0 - p["f_XI"]) * e["X_S"]
    nu[11] = decay_h - e["X_H"]
    nu[12] = -e["S_A"] - p["Y_PO4"] * e["X_PP"] + e["X_PHA"]
    nu[13] = -ypha * e["S_O"] + e["X_PP"] - ypha * e["X_PHA"]
    nu[14] = beta_s * (e["S_NO2"] - e["S_NO3"]) + e["X_PP"] - ypha * e["X_PHA"]
    nu[15] = gamma_s * (e["S_N2"] - e["S_NO2"]) + e["X_PP"] - ypha * e["X_PHA"]
    nu[16] = (1.0 - 1.0 / ypao) * e["S_O"] + e["X_PAO"] - 1.0 / ypao * e["X_PHA"]
    nu[17] = beta_p * (e["S_NO2"] - e["S_NO3"]) + e["X_PAO"] - 1.0 / ypao * e["X_PHA"]
    nu[18] = gamma_p * (e["S_N2"] - e["S_NO2"]) + e["X_PAO"] - 1.0 / ypao * e["X_PHA"]
    nu[19] = decay_h - e["X_PAO"]
    nu[20] = -e["X_PP"]
    nu[21] = e["S_A"] - e["X_PHA"]
    nu[22] = (
        -(3.43 - p["Y_AOB"]) / p["Y_AOB"] * e["S_O"]
        + 1.0 / p["Y_AOB"] * e["S_NO2"]
        + e["X_AOB"]
    )
    nu[23] = (
        -(1.14 - p["Y_NOB"]) / p["Y_NOB"] * e["S_O"]
        - 1.0 / p["Y_NOB"] * e["S_NO2"]
        + 1.0 / p["Y_NOB"] * e["S_NO3"]
        + e["X_NOB"]
    )
    nu[24] = decay_h - e["X_AOB"]
    nu[25] = decay_h - e["X_NOB"]
    nu[26] = -e["S_PO4"] - p["gamma_OH"] * e["X_MeOH"] + 1.0 / p["i_PMeP"] * e["X_MeP"]
    nu[27] = e["S_PO4"] + p["gamma_OH"] * e["X_MeOH"] - 1.0 / p["i_PMeP"] * e["X_MeP"]
    n_weights = {
        "S_F": p["i_NSF"],
        "S_I": p["i_NSI"],
        "S_N2": 1.0,
        "S_NO2": 1.0,
        "S_NO3": 1.0,
        "X_I": p["i_NXI"],
        "X_S": p["i_NXS"],
        "X_H": p["i_NBM"],
        "X_PAO": p["i_NBM"],
        "X_AOB": p["i_NBM"],
        "X_NOB": p["i_NBM"],
    }
    nu[:, COMPONENT_INDEX["S_NH4"]] = -sum(
        (weight * nu[:, COMPONENT_INDEX[name]] for name, weight in n_weights.items())
    )
    p_weights = {
        "S_F": p["i_PSF"],
        "S_I": p["i_PSI"],
        "X_I": p["i_PXI"],
        "X_S": p["i_PXS"],
        "X_H": p["i_PBM"],
        "X_PAO": p["i_PBM"],
        "X_AOB": p["i_PBM"],
        "X_NOB": p["i_PBM"],
        "X_PP": 1.0,
        "X_MeP": p["i_PMeP"],
    }
    phosphorus = -sum(
        (weight * nu[:, COMPONENT_INDEX[name]] for name, weight in p_weights.items())
    )
    nu[:26, COMPONENT_INDEX["S_PO4"]] = phosphorus[:26]
    nu[:, COMPONENT_INDEX["S_ALK"]] = (
        nu[:, COMPONENT_INDEX["S_NH4"]] / 14.0
        - nu[:, COMPONENT_INDEX["S_NO2"]] / 14.0
        - nu[:, COMPONENT_INDEX["S_NO3"]] / 14.0
        + nu[:, COMPONENT_INDEX["S_PO4"]] / 31.0
    )
    return nu


STOICHIOMETRIC_MATRIX = build_stoichiometric_matrix()


def build_invariant_matrix(
    parameters: Mapping[str, float] = PARAMETERS, *, normalized: bool = True
) -> FloatArray:
    """Return the five named reaction- and aeration-invariant inventory rows."""
    p = parameters
    e = {name: _e(name) for name in COMPONENTS}
    b_si = e["S_I"]
    b_n = (
        e["S_NH4"]
        + e["S_NO2"]
        + e["S_NO3"]
        + e["S_N2"]
        + p["i_NSI"] * e["S_I"]
        + p["i_NSF"] * e["S_F"]
        + p["i_NXI"] * e["X_I"]
        + p["i_NXS"] * e["X_S"]
        + p["i_NBM"] * (e["X_H"] + e["X_PAO"] + e["X_AOB"] + e["X_NOB"])
    )
    b_p = (
        e["S_PO4"]
        + p["i_PSI"] * e["S_I"]
        + p["i_PSF"] * e["S_F"]
        + p["i_PXI"] * e["X_I"]
        + p["i_PXS"] * e["X_S"]
        + e["X_PP"]
        + p["i_PBM"] * (e["X_H"] + e["X_PAO"] + e["X_AOB"] + e["X_NOB"])
        + p["i_PMeP"] * e["X_MeP"]
    )
    b_alk = (
        e["S_ALK"]
        - e["S_NH4"] / 14.0
        + e["S_NO2"] / 14.0
        + e["S_NO3"] / 14.0
        - e["S_PO4"] / 31.0
    )
    b_metal = p["gamma_OH"] * e["X_MeP"] + 1.0 / p["i_PMeP"] * e["X_MeOH"]
    matrix = np.vstack((b_si, b_n, b_p, b_alk, b_metal))
    if normalized:
        matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix


INVARIANT_MATRIX = build_invariant_matrix()
