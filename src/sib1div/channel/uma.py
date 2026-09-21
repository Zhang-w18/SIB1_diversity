"""Sionna 1.0.2 UMa drop generation and immediate TXRU-domain projection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from sib1div.codebook.array import build_txru_mapping
from sib1div.config import SimulationConfig


@dataclass(frozen=True)
class UMaDrop:
    drop_index: int
    ue_position_m: NDArray[np.float64]
    radius_m: float
    azimuth_deg: float
    los: bool
    path_coefficients: NDArray[np.complex128]  # [rx,dual-pol TXRU,path]
    path_delays_s: NDArray[np.float64]
    cluster_powers: NDArray[np.float64]
    pathloss_db: float
    large_scale_amplitude_gain: float
    large_scale_power_gain: float
    path_spatial_covariances: NDArray[np.complex128]  # [rx,path,txru,txru]

    def frequency_response(self, n_subcarriers: int, spacing_hz: float) -> NDArray[np.complex128]:
        offsets = (np.arange(n_subcarriers) - 0.5 * (n_subcarriers - 1)) * spacing_hz
        phase = np.exp(-1j * 2 * np.pi * offsets[:, None] * self.path_delays_s[None, :])
        return np.einsum("rtl,kl->rkt", self.path_coefficients, phase, optimize=True)


def sample_ue_position(config: SimulationConfig, drop_index: int) -> tuple[NDArray[np.float64], float, float]:
    scenario = config.data["scenario"]
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, 0x554D41, drop_index]))
    r_min = float(scenario["ue_distance_min_m"])
    r_max = float(scenario["ue_distance_max_m"])
    radius = float(np.sqrt(r_min**2 + rng.random() * (r_max**2 - r_min**2)))
    azimuth = float(rng.uniform(
        float(scenario["sector_azimuth_min_deg"]),
        float(scenario["sector_azimuth_max_deg"]),
    ))
    phi = np.deg2rad(azimuth)
    position = np.array([
        radius * np.cos(phi), radius * np.sin(phi), float(scenario["ue_height_m"])
    ])
    return position, radius, azimuth


class UMaChannel:
    """One-BS/one-UE downlink UMa channel with transient AE coefficients."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        data = config.data
        array = data["bs_array"]
        ue = data["ue_array"]
        carrier = float(data["scenario"]["carrier_frequency_hz"])
        from sionna.phy.channel.tr38901 import PanelArray, UMa

        self._bs_array = PanelArray(
            num_rows_per_panel=int(array["vertical_aes"]),
            num_cols_per_panel=int(array["horizontal_aes"]),
            polarization="dual",
            polarization_type="cross",
            antenna_pattern="38.901",
            carrier_frequency=carrier,
            element_vertical_spacing=float(array["vertical_spacing_lambda"]),
            element_horizontal_spacing=float(array["horizontal_spacing_lambda"]),
        )
        self._ut_array = PanelArray(
            num_rows_per_panel=int(ue["vertical_elements"]),
            num_cols_per_panel=int(ue["horizontal_elements"]),
            polarization="dual",
            polarization_type="cross",
            antenna_pattern="omni",
            carrier_frequency=carrier,
            element_vertical_spacing=0.5,
            element_horizontal_spacing=0.5,
        )
        self._model = UMa(
            carrier, "low", self._ut_array, self._bs_array, "downlink",
            enable_pathloss=True, enable_shadow_fading=True,
            always_generate_lsp=True,
        )
        self._model.return_rays = True
        single = build_txru_mapping(
            int(array["vertical_aes"]), int(array["horizontal_aes"]),
            int(array["vertical_txrus_per_pol"]), int(array["horizontal_txrus_per_pol"]),
        )
        # Sionna orders each polarization horizontal-major, while the platform
        # codebook uses vertical-major. Reorder AE rows before projection.
        m = int(array["vertical_aes"])
        n = int(array["horizontal_aes"])
        self._sionna_to_platform = np.array([col * m + row for row in range(m) for col in range(n)])
        self._mapping = np.zeros((2 * single.shape[0], 2 * single.shape[1]), dtype=np.complex128)
        self._mapping[:single.shape[0], :single.shape[1]] = single
        self._mapping[single.shape[0]:, single.shape[1]:] = single

    def _sample_cir_and_rays(self):
        """Run Sionna's frozen 38.901 stages while retaining their LSPs.

        The public system-level call returns rays but discards the sampled
        Rician K factor needed to form the conditional long-term covariance.
        Keeping it here avoids substituting instantaneous |h_l|^2 for path
        power.  These private calls are intentionally tied to Sionna 1.0.2,
        which is recorded in every run's provenance.
        """
        import tensorflow as tf
        from sionna.phy.channel.tr38901 import Topology
        from sionna.phy.channel.utils import deg_2_rad

        model = self._model
        lsp = model._lsp_sampler()  # pylint: disable=protected-access
        rays = model._ray_sampler(lsp)  # pylint: disable=protected-access
        scenario = model._scenario  # pylint: disable=protected-access
        topology = Topology(
            velocities=scenario.ut_velocities,
            moving_end="rx",
            los_aoa=deg_2_rad(scenario.los_aoa),
            los_aod=deg_2_rad(scenario.los_aod),
            los_zoa=deg_2_rad(scenario.los_zoa),
            los_zod=deg_2_rad(scenario.los_zod),
            los=scenario.los,
            distance_3d=scenario.distance_3d,
            tx_orientations=scenario.bs_orientations,
            rx_orientations=scenario.ut_orientations,
        )
        c_ds = scenario.get_param("cDS") * 1e-9
        h_unscaled, delays = model._cir_sampler(  # pylint: disable=protected-access
            1, 1.0, lsp.k_factor, rays, topology, c_ds,
        )
        h_scaled = model._step_12(h_unscaled, lsp.sf)  # pylint: disable=protected-access
        numerator = tf.reduce_sum(tf.abs(h_scaled) ** 2)
        denominator = tf.reduce_sum(tf.abs(h_unscaled) ** 2)
        large_scale_gain = float(tf.sqrt(numerator / denominator).numpy())
        # Match SystemLevelChannel.__call__ output ordering.
        h_scaled = tf.transpose(h_scaled, [0, 2, 4, 1, 5, 3, 6])
        delays = tf.transpose(delays, [0, 2, 1, 3])
        return h_scaled, delays, rays, topology, lsp.k_factor, c_ds, large_scale_gain

    def _long_term_path_covariances(
        self, rays, topology, k_factor, c_ds, output_delays, large_scale_gain: float,
    ) -> NDArray[np.complex128]:
        """Analytically sum independent 38.901 ray/polarization moments.

        Random initial phases are integrated out by summing four polarization
        basis outer products.  Strong-cluster sub-rays are assigned to the
        same three delayed subclusters as Sionna's CIR generator.  No channel
        realizations are averaged and no instantaneous coefficient magnitude
        is used as a path power.
        """
        import tensorflow as tf

        generator = self._model._cir_sampler  # pylint: disable=protected-access
        powers = np.asarray(rays.powers.numpy()[0, 0, 0], dtype=float)
        delays = np.asarray(rays.delays.numpy()[0, 0, 0], dtype=float)
        xpr = np.asarray(rays.xpr.numpy(), dtype=float)
        n_rays = xpr.shape[-1]
        strongest = np.argsort(-powers)
        ray_groups = (
            np.asarray(generator._sub_cl_1_ind, dtype=int),  # pylint: disable=protected-access
            np.asarray(generator._sub_cl_2_ind, dtype=int),  # pylint: disable=protected-access
            np.asarray(generator._sub_cl_3_ind, dtype=int),  # pylint: disable=protected-access
        )
        offsets = np.asarray(generator._sub_cl_delay_offsets, dtype=float)  # pylint: disable=protected-access
        c_ds_value = float(np.asarray(c_ds.numpy()).reshape(-1)[0])
        entries: list[tuple[float, int, NDArray[np.int64]]] = []
        for group, offset in zip(ray_groups, offsets, strict=True):
            for cluster in strongest[:2]:
                entries.append((delays[cluster] + offset * c_ds_value, int(cluster), group))
        all_rays = np.arange(n_rays, dtype=int)
        for cluster in strongest[2:]:
            entries.append((delays[cluster], int(cluster), all_rays))
        entries.sort(key=lambda item: item[0])
        expected_delays = np.array([item[0] for item in entries])
        if not np.allclose(expected_delays, output_delays, rtol=0.0, atol=1e-12):
            raise RuntimeError("Sionna path grouping changed; covariance delays no longer match the CIR")

        h_array = generator._step_11_array_offsets(  # pylint: disable=protected-access
            topology, rays.aoa, rays.aod, rays.zoa, rays.zod,
        )
        n_rx = int(h_array.shape[-2])
        n_txru = self._mapping.shape[1]
        covariance = np.zeros(
            (n_rx, len(entries), n_txru, n_txru), dtype=np.complex128,
        )
        entry_for_ray: dict[tuple[int, int], int] = {}
        for path, (_, cluster, group) in enumerate(entries):
            for ray in group:
                entry_for_ray[(cluster, int(ray))] = path

        # Four independently phased entries of the 2x2 polarization matrix.
        for polarization_term, (row, col) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
            phase_matrix = np.zeros(xpr.shape + (2, 2), dtype=np.complex64)
            amplitude = np.ones_like(xpr)
            if polarization_term in (1, 2):
                amplitude = 1.0 / np.sqrt(xpr)
            phase_matrix[..., row, col] = amplitude
            h_field = generator._step_11_field_matrix(  # pylint: disable=protected-access
                topology, rays.aoa, rays.aod, rays.zoa, rays.zod,
                tf.constant(phase_matrix),
            )
            components = np.asarray((h_field * h_array).numpy()[0, 0, 0], dtype=np.complex128)
            components *= np.sqrt(powers[:, None, None, None] / n_rays)
            if bool(np.asarray(topology.los.numpy()).reshape(-1)[0]):
                k = float(np.asarray(k_factor.numpy()).reshape(-1)[0])
                components /= np.sqrt(k + 1.0)
            components *= large_scale_gain
            # [cluster,ray,rx,AE] -> platform AE order -> TXRU
            components = components[..., self._sionna_to_platform.tolist()
                + (components.shape[-1] // 2 + self._sionna_to_platform).tolist()]
            components = np.einsum("cmra,at->cmrt", components, self._mapping, optimize=True)
            for (cluster, ray), path in entry_for_ray.items():
                vectors = components[cluster, ray]
                covariance[:, path] += np.einsum("rt,ru->rtu", vectors, vectors.conj())

        if bool(np.asarray(topology.los.numpy()).reshape(-1)[0]):
            k = float(np.asarray(k_factor.numpy()).reshape(-1)[0])
            los = np.asarray(generator._step_11_los(  # pylint: disable=protected-access
                topology, tf.constant([0.0], generator.rdtype),
            ).numpy()[0, 0, 0, 0, :, :, 0], dtype=np.complex128)
            per_pol = los.shape[-1] // 2
            order = np.r_[self._sionna_to_platform, per_pol + self._sionna_to_platform]
            los = los[:, order] @ self._mapping
            los *= np.sqrt(k / (k + 1.0)) * large_scale_gain
            covariance[:, 0] += np.einsum("rt,ru->rtu", los, los.conj())

        return 0.5 * (covariance + covariance.swapaxes(-1, -2).conj())

    def generate(self, drop_index: int) -> UMaDrop:
        import tensorflow as tf
        from sionna.phy import config as sionna_config

        position, radius, azimuth = sample_ue_position(self.config, drop_index)
        data = self.config.data
        scenario = data["scenario"]
        panel = data["panel"]
        derived_seed = int(np.random.SeedSequence([self.config.seed, 0x53494F, drop_index]).generate_state(1)[0])
        sionna_config.seed = derived_seed
        z = tf.zeros([1, 1, 3], tf.float32)
        bs_orientation = np.deg2rad([
            float(panel["azimuth_deg"]),
            float(panel["mechanical_downtilt_deg"]),
            0.0,
        ]).reshape(1, 1, 3)
        self._model.set_topology(
            ut_loc=tf.constant(position.reshape(1, 1, 3), tf.float32),
            bs_loc=tf.constant([[[0.0, 0.0, float(scenario["bs_height_m"])]]], tf.float32),
            ut_orientations=z,
            bs_orientations=tf.constant(bs_orientation, tf.float32),
            ut_velocities=z,
            in_state=tf.constant([[False]]),
            los=None,
        )
        h, delays, rays, topology, k_factor, c_ds, gain = self._sample_cir_and_rays()
        h_ae = np.asarray(h.numpy()[0, 0, :, 0, :, :, 0], dtype=np.complex128)
        per_pol = h_ae.shape[1] // 2
        order = np.r_[self._sionna_to_platform, per_pol + self._sionna_to_platform]
        h_ae = h_ae[:, order, :]
        h_txru = np.einsum("ral,at->rtl", h_ae, self._mapping, optimize=True)
        tau = np.asarray(delays.numpy()[0, 0, 0], dtype=float)
        path_covariances = self._long_term_path_covariances(
            rays, topology, k_factor, c_ds, tau, gain,
        )
        cluster_powers = np.asarray(rays.powers.numpy()[0, 0, 0], dtype=float)
        los = bool(self._model._scenario.los.numpy()[0, 0, 0])
        pathloss = float(self._model._scenario.basic_pathloss.numpy()[0, 0, 0])
        return UMaDrop(
            drop_index=drop_index,
            ue_position_m=position,
            radius_m=radius,
            azimuth_deg=azimuth,
            los=los,
            path_coefficients=h_txru,
            path_delays_s=tau,
            cluster_powers=cluster_powers,
            pathloss_db=pathloss,
            large_scale_amplitude_gain=gain,
            large_scale_power_gain=gain**2,
            path_spatial_covariances=path_covariances,
        )
