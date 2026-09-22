"""Fixed-statistics 3GPP CDL channel used by v4 section 2.8 mode C."""

from __future__ import annotations

import numpy as np

from sib1div.codebook.array import build_txru_mapping
from sib1div.config import SimulationConfig
from sib1div.schemes import dual_polarized

from .uma import UMaDrop


class FixedCDLChannel:
    """CDL generator whose rays, coupling, arrays, and velocity stay frozen.

    Only the CIR generator's random initial phases are resampled per
    realization.  A disjoint seed namespace is used to Monte-Carlo estimate
    the transmit covariance and the common reference receive power before the
    BLER stream starts.
    """

    def __init__(self, config: SimulationConfig, ssb_weights: np.ndarray):
        import tensorflow as tf
        from sionna.phy.channel.tr38901 import CDL, PanelArray, Rays, Topology
        from sionna.phy import config as sionna_config

        self.config = config
        data = config.data
        settings = data["fixed_cdl_statistics"]
        array = data["bs_array"]
        ue = data["ue_array"]
        carrier = float(data["scenario"]["carrier_frequency_hz"])
        self._tf = tf
        self._sionna_config = sionna_config
        self._statistics_seed = int(settings["statistics_seed"])
        self._realization_seed = int(settings["realization_seed"])
        self._time_steps = int(settings.get("time_steps_per_slot", 1))
        self._time_step_s = float(settings.get("time_step_s", 0.0))
        if self._time_steps > 1 and self._time_step_s <= 0.0:
            raise ValueError("fixed CDL time_step_s must be positive when time_steps_per_slot > 1")

        self._bs_array = PanelArray(
            num_rows_per_panel=int(array["vertical_aes"]),
            num_cols_per_panel=int(array["horizontal_aes"]), polarization="dual",
            polarization_type="cross", antenna_pattern="38.901",
            carrier_frequency=carrier,
            element_vertical_spacing=float(array["vertical_spacing_lambda"]),
            element_horizontal_spacing=float(array["horizontal_spacing_lambda"]),
        )
        self._ut_array = PanelArray(
            num_rows_per_panel=int(ue["vertical_elements"]),
            num_cols_per_panel=int(ue["horizontal_elements"]), polarization="dual",
            polarization_type="cross", antenna_pattern="omni",
            carrier_frequency=carrier, element_vertical_spacing=0.5,
            element_horizontal_spacing=0.5,
        )
        panel = data["panel"]
        self._model = CDL(
            str(settings["profile"]).upper(), float(settings["delay_spread_s"]), carrier,
            self._ut_array, self._bs_array, "downlink",
            ut_orientation=tf.constant(np.deg2rad(settings.get("ue_orientation_deg", [180, 0, 0])), tf.float32),
            bs_orientation=tf.constant(np.deg2rad([
                float(panel["azimuth_deg"]), float(panel["mechanical_downtilt_deg"]), 0.0,
            ]), tf.float32),
            min_speed=float(settings.get("ue_speed_mps", 0.0)),
            max_speed=float(settings.get("ue_speed_mps", 0.0)),
        )
        self.angle_statistics = self._apply_angle_transform(
            settings.get("angle_transform", {})
        )

        single = build_txru_mapping(
            int(array["vertical_aes"]), int(array["horizontal_aes"]),
            int(array["vertical_txrus_per_pol"]), int(array["horizontal_txrus_per_pol"]),
        )
        m, n = int(array["vertical_aes"]), int(array["horizontal_aes"])
        self._order = np.array([col * m + row for row in range(m) for col in range(n)])
        self._mapping = np.zeros((2 * single.shape[0], 2 * single.shape[1]), complex)
        self._mapping[:single.shape[0], :single.shape[1]] = single
        self._mapping[single.shape[0]:, single.shape[1]:] = single

        # Freeze velocity direction and CDL ray coupling in the statistics stream.
        sionna_config.seed = self._statistics_seed
        batch = 1
        speed = float(settings.get("ue_speed_mps", 0.0))
        velocity_az = np.deg2rad(float(settings.get("velocity_azimuth_deg", 0.0)))
        velocity_el = np.deg2rad(float(settings.get("velocity_elevation_deg", 0.0)))
        velocity = [speed*np.cos(velocity_el)*np.cos(velocity_az),
                    speed*np.cos(velocity_el)*np.sin(velocity_az), speed*np.sin(velocity_el)]
        tile3 = lambda value: tf.tile(value, [batch, 1, 1])
        self._topology = Topology(
            velocities=tf.constant([[velocity]], self._model.rdtype), moving_end="rx",
            los_aoa=tile3(self._model._los_aoa), los_aod=tile3(self._model._los_aod),
            los_zoa=tile3(self._model._los_zoa), los_zod=tile3(self._model._los_zod),
            los=tf.fill([batch, 1, 1], self._model._los),
            distance_3d=tf.zeros([batch, 1, 1], self._model.rdtype),
            tx_orientations=tf.tile(tf.reshape(self._model._tx_orientation, [1, 1, 3]), [batch, 1, 1]),
            rx_orientations=tf.tile(tf.reshape(self._model._rx_orientation, [1, 1, 3]), [batch, 1, 1]),
        )
        def tiled(value, multiples): return tf.tile(value, multiples)
        aoa, aod, zoa, zod = self._model._random_coupling(
            tiled(self._model._aoa, [1,1,1,1,1]), tiled(self._model._aod, [1,1,1,1,1]),
            tiled(self._model._zoa, [1,1,1,1,1]), tiled(self._model._zod, [1,1,1,1,1]))
        self._rays = Rays(
            delays=tiled(self._model._delays * self._model._delay_spread, [1,1,1,1]),
            powers=tiled(self._model._powers, [1,1,1,1]), aoa=aoa, aod=aod,
            zoa=zoa, zod=zod, xpr=tiled(self._model._xpr, [1,1,1,1,1]),
        )
        self._k_factor = tiled(self._model._k_factor, [1, 1, 1])
        self._delays = np.asarray(self._rays.delays.numpy()[0, 0, 0], float)
        self._powers = np.asarray(self._rays.powers.numpy()[0, 0, 0], float)
        self._ssb_weights = np.asarray(ssb_weights)
        self._prepare_long_term_reference(int(settings["covariance_realizations"]))

    def _apply_angle_transform(self, transform: dict) -> dict[str, dict[str, float]]:
        """Apply power-weighted CDL ray-angle centering/scaling before coupling."""
        tf = self._tf
        cluster_powers = np.asarray(self._model._powers.numpy(), float).reshape(-1)
        spread_keys = {
            "aod": "target_asd_deg", "aoa": "target_asa_deg",
            "zod": "target_zsd_deg", "zoa": "target_zsa_deg",
        }
        statistics: dict[str, dict[str, float]] = {}
        for name in ("aod", "aoa", "zod", "zoa"):
            mean_key, scale_key = f"mean_{name}_deg", f"{name}_scale"
            internal = np.rad2deg(np.asarray(
                getattr(self._model, f"_{name}").numpy(), float,
            ))
            # The project codebook uses global phi positive in the opposite
            # direction from Sionna's CDL departure azimuth. Keep the public
            # mode-C AoD fields in the project/codebook convention.
            values = -internal if name == "aod" else internal
            values = values.reshape(cluster_powers.size, -1)
            weights = np.broadcast_to(
                cluster_powers[:, None] / values.shape[1], values.shape,
            )
            if name in {"aod", "aoa"}:
                old_mean = float(np.rad2deg(np.angle(np.sum(
                    weights * np.exp(1j * np.deg2rad(values))
                ))))
                offsets = (values - old_mean + 180.0) % 360.0 - 180.0
            else:
                old_mean = float(np.sum(weights * values) / np.sum(weights))
                offsets = values - old_mean
            old_spread = float(np.sqrt(np.sum(weights * offsets**2) / np.sum(weights)))
            target = float(transform.get(mean_key, old_mean))
            spread_key = spread_keys[name]
            if spread_key in transform:
                scale = float(transform[spread_key]) / old_spread
            else:
                scale = float(transform.get(scale_key, 1.0))
            shifted = target + scale * offsets
            if name in {"aod", "aoa"}:
                target_spread = transform.get(spread_key)
                for _ in range(8):
                    shifted = (shifted + 180.0) % 360.0 - 180.0
                    provisional_mean = float(np.rad2deg(np.angle(np.sum(
                        weights * np.exp(1j * np.deg2rad(shifted))
                    ))))
                    centered = (shifted - provisional_mean + 180.0) % 360.0 - 180.0
                    if target_spread is not None:
                        provisional_spread = float(np.sqrt(
                            np.sum(weights * centered**2) / np.sum(weights)
                        ))
                        centered *= float(target_spread) / provisional_spread
                    shifted = target + centered
            else:
                shifted = np.clip(shifted, 0.0, 180.0)
            stored = -shifted if name == "aod" else shifted
            setattr(
                self._model, f"_{name}",
                tf.constant(np.deg2rad(stored.reshape(
                    np.asarray(getattr(self._model, f"_{name}").numpy()).shape
                )), self._model.rdtype),
            )
            if name in {"aod", "aoa"}:
                new_mean = float(np.rad2deg(np.angle(np.sum(
                    weights * np.exp(1j * np.deg2rad(shifted))
                ))))
                new_offsets = (shifted - new_mean + 180.0) % 360.0 - 180.0
            else:
                new_mean = float(np.sum(weights * shifted) / np.sum(weights))
                new_offsets = shifted - new_mean
            new_spread = float(np.sqrt(
                np.sum(weights * new_offsets**2) / np.sum(weights)
            ))
            statistics[name] = {
                "original_mean_deg": old_mean,
                "original_rms_spread_deg": old_spread,
                "applied_scale": scale,
                "transformed_mean_deg": new_mean,
                "transformed_rms_spread_deg": new_spread,
                "sionna_internal_transformed_mean_deg": (
                    -new_mean if name == "aod" else new_mean
                ),
            }
        return statistics

    def long_term_beam_powers(self, single_polarization_weights: np.ndarray) -> np.ndarray:
        """Return long-term receive powers for single-polarization TXRU beams."""
        weights = np.asarray(single_polarization_weights)
        beams = np.column_stack([
            dual_polarized(weights[:, index]) for index in range(weights.shape[1])
        ])
        return np.einsum(
            "tb,tu,ub->b", beams.conj(), self.transmit_covariance, beams,
        ).real

    def _coefficients(self, seed: int) -> np.ndarray:
        self._sionna_config.seed = int(seed)
        sampling_frequency = 1.0 if self._time_steps == 1 else 1.0 / self._time_step_s
        h, _ = self._model._cir_sampler(
            self._time_steps, sampling_frequency, self._k_factor, self._rays, self._topology,
        )
        h = self._tf.transpose(h, [0, 2, 4, 1, 5, 3, 6])
        # [rx,AE,path,time] -> platform AE order -> [time,rx,txru,path]
        ae = np.asarray(h.numpy()[0, 0, :, 0, :, :, :], complex)
        per_pol = ae.shape[1] // 2
        order = np.r_[self._order, per_pol + self._order]
        return np.einsum("raln,at->nrtl", ae[:, order, :, :], self._mapping, optimize=True)

    def _response(self, coefficients: np.ndarray) -> np.ndarray:
        n_sc = self.config.active_subcarriers
        spacing = float(self.config.data["nr"]["subcarrier_spacing_hz"])
        offsets = (np.arange(n_sc) - .5 * (n_sc - 1)) * spacing
        phase = np.exp(-1j * 2*np.pi * offsets[:, None] * self._delays[None, :])
        return np.einsum("nrtl,kl->nrkt", coefficients, phase, optimize=True)

    def _prepare_long_term_reference(self, count: int) -> None:
        if count < 1:
            raise ValueError("fixed CDL covariance_realizations must be positive")
        n_rx = int(self._ut_array.num_ant)
        n_tx = self._mapping.shape[1]
        n_path = self._delays.size
        path_cov = np.zeros((n_rx, n_path, n_tx, n_tx), complex)
        rt = np.zeros((n_tx, n_tx), complex)
        for index in range(count):
            coeff = self._coefficients(int(np.random.SeedSequence(
                [self._statistics_seed, 0x434F56, index]).generate_state(1)[0]))
            path_cov += np.einsum("nrtl,nrul->rltu", coeff, coeff.conj()) / self._time_steps
            response = self._response(coeff)
            # Sum over Rx branches to match ||H[k]w||^2 in the mode-C SNR
            # definition; average only over realization, time, and frequency.
            rt += np.einsum("nrkt,nrku->tu", response.conj(), response) / (self._time_steps * self.config.active_subcarriers)
        self.path_spatial_covariances = path_cov / count
        self.transmit_covariance = 0.5 * (rt/count + (rt/count).conj().T)
        powers = self.long_term_beam_powers(self._ssb_weights)
        self.selected_ssb = int(np.argmax(powers))
        self.ssb_long_term_powers = powers
        self.reference_receive_power = float(powers[self.selected_ssb])

    def generate(self, drop_index: int) -> UMaDrop:
        seed = int(np.random.SeedSequence(
            [self._realization_seed, 0x43444C, drop_index]).generate_state(1)[0])
        coefficients = self._coefficients(seed)
        stored = coefficients[0] if self._time_steps == 1 else coefficients
        return UMaDrop(
            drop_index=drop_index, ue_position_m=np.zeros(3), radius_m=0.0,
            azimuth_deg=float(self.config.data["fixed_cdl_statistics"].get("angle_transform", {}).get("mean_aod_deg", 0.0)),
            los=bool(np.asarray(self._model._los.numpy())), path_coefficients=stored,
            path_delays_s=self._delays.copy(), cluster_powers=self._powers.copy(),
            pathloss_db=0.0, large_scale_amplitude_gain=1.0, large_scale_power_gain=1.0,
            path_spatial_covariances=self.path_spatial_covariances.copy(),
            fixed_selected_ssb=self.selected_ssb,
            reference_receive_power=self.reference_receive_power,
            channel_time_step_s=self._time_step_s,
        )
