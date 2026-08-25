"""Known-answer tests for verified model-frame access and preprocessing."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from tcv_diagnostics.model_training_data import (
    CodecFrameDataset,
    FAMILY_FIELDS,
    ModelDatasetCatalog,
    OFFICIAL_DATASET_RESULT_SHA256,
    OFFICIAL_NORMALIZATION_SHA256,
    VOLUME_SHAPE,
    _strict_contiguous_frames,
    epoch_order,
    toroidal_roll,
)
from tcv_diagnostics.o2_context_data import OneStepContextDataset
from tcv_diagnostics.o2_training_data import (
    OneStepWindowDataset,
    strict_o2_targets,
)


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalization_record() -> dict:
    names = set().union(*map(set, FAMILY_FIELDS.values()))
    records = {}
    for name in names:
        records[name] = {
            "mean": 0.0 if name == "Ne" else 1.0,
            "population_standard_deviation": 2.0,
            "transform": (
                {"name": "log_offset", "offset": 1.0e-6}
                if name == "Ne"
                else {"name": "identity"}
            ),
        }
    records["Bphi/inner"] = {
        "mean": 1.0,
        "population_standard_deviation": 2.0,
        "transform": {"name": "identity"},
    }
    records["Bphi/outer"] = {
        "mean": 2.0,
        "population_standard_deviation": 4.0,
        "transform": {"name": "identity"},
    }
    return {
        "development_run": "85604",
        "held_out_85606_read": False,
        "fit_frames": [0, 432],
        "records": records,
    }


def _create_sparse_shards(root: Path) -> dict:
    shard_records = []
    fields = ("Ne", "Pe", "Pi", "NVe", "NVi", "Vort", "phi", "Vi")
    for index in range(8):
        start = index * 78
        stop = start + 78
        path = root / "shards" / f"model_dataset_shard_{index:03d}.h5"
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "x") as handle:
            handle.attrs["development_run"] = "85604"
            handle.attrs["held_out_85606_read"] = False
            handle.attrs["zperiod"] = 5
            handle.attrs["global_start_inclusive"] = start
            handle.attrs["global_stop_exclusive"] = stop
            group = handle.create_group("fields")
            for field in fields:
                group.create_dataset(
                    field,
                    shape=(78, *VOLUME_SHAPE),
                    dtype="f4",
                    chunks=(1, *VOLUME_SHAPE),
                )
            boundary = handle.create_group("boundary")
            boundary.create_dataset(
                "Bphi", shape=(78, 2, 32), dtype="f4", chunks=(1, 2, 32)
            )
            coordinates = handle.create_group("coordinates")
            coordinates.create_dataset(
                "frame_index", data=np.arange(start, stop, dtype=np.int64)
            )
            coordinates.create_dataset(
                "time", data=np.arange(start, stop, dtype=np.float64) * 300.0
            )

            if index == 0:
                z = np.arange(88, dtype=np.float32)[None, None, :]
                base = np.broadcast_to(z, VOLUME_SHAPE).copy()
                for channel, field in enumerate(fields):
                    values = base + np.float32(channel + 2)
                    group[field][0] = values
                    group[field][1] = values + np.float32(10)
                    group[field][2] = values + np.float32(20)
                boundary["Bphi"][0, 0] = np.arange(32, dtype=np.float32) + 1
                boundary["Bphi"][0, 1] = np.arange(32, dtype=np.float32) + 2
            if index == 6:
                local = 496 - start
                for channel, field in enumerate(fields):
                    group[field][local] = np.float32(channel + 3)
                    group[field][local + 1] = np.float32(channel + 13)
                    group[field][local + 2] = np.float32(channel + 23)
                boundary["Bphi"][local, 0] = np.arange(32, dtype=np.float32) + 3
                boundary["Bphi"][local, 1] = np.arange(32, dtype=np.float32) + 6

        shard_records.append(
            {
                "index": index,
                "global_start_inclusive": start,
                "global_stop_exclusive": stop,
                "path": f"/official/shards/{path.name}",
                "sha256": _sha256(path),
            }
        )
    return {
        "development_run": "85604",
        "held_out_85606_read": False,
        "training_performed": False,
        "gates": {"all_passed": True},
        "dataset": {
            "fields": ["Ne", "Pe", "Pi", "NVe", "NVi", "Vort", "phi", "Vi"],
            "frame_count": 624,
            "shards": shard_records,
        },
    }


class TestModelTrainingData(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "development_artifacts"
        self.record = _create_sparse_shards(self.root)
        self.catalog = ModelDatasetCatalog(
            self.record,
            _normalization_record(),
            artifact_root=self.root,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_official_tracked_record_hashes_are_locked(self) -> None:
        self.assertEqual(
            _sha256(ROOT / "paper0/results/phase2_model_dataset_6893525.json"),
            OFFICIAL_DATASET_RESULT_SHA256,
        )
        self.assertEqual(
            _sha256(
                ROOT
                / "paper0/results/phase2_model_dataset_normalization_6893525.json"
            ),
            OFFICIAL_NORMALIZATION_SHA256,
        )

    def test_c5p_field_order_and_frozen_normalization(self) -> None:
        dataset = CodecFrameDataset(
            self.catalog,
            family="c5p",
            split="train",
            frames=[0],
            augment=False,
            seed=1701,
        )
        item = dataset[0]

        self.assertEqual(tuple(item["volume"].shape), (5, *VOLUME_SHAPE))
        expected_ne = np.log(np.float64(2.0) + 1.0e-6) / 2.0
        self.assertAlmostEqual(float(item["volume"][0, 0, 0, 0]), expected_ne, 6)
        self.assertEqual(float(item["volume"][1, 0, 0, 0]), 1.0)
        self.assertEqual(int(item["frame_index"]), 0)
        self.assertEqual(int(item["toroidal_roll"]), 0)
        self.assertNotIn("time", item)
        self.assertNotIn("boundary", item)
        dataset.close()

    def test_e6b_boundary_sides_are_normalized_independently(self) -> None:
        dataset = CodecFrameDataset(
            self.catalog,
            family="e6b",
            split="validation",
            frames=[496],
            augment=False,
            seed=1701,
            return_physical=True,
        )
        item = dataset[0]

        self.assertEqual(tuple(item["volume"].shape), (6, *VOLUME_SHAPE))
        self.assertEqual(tuple(item["boundary"].shape), (2, 32))
        self.assertEqual(float(item["boundary"][0, 0]), 1.0)
        self.assertEqual(float(item["boundary"][1, 0]), 1.0)
        self.assertEqual(tuple(item["physical_volume"].shape), (6, *VOLUME_SHAPE))
        self.assertEqual(tuple(item["physical_boundary"].shape), (2, 32))
        self.assertEqual(float(item["physical_boundary"][0, 0]), 3.0)
        self.assertEqual(float(item["physical_boundary"][1, 0]), 6.0)
        self.assertEqual(int(item["frame_index"]), 496)
        dataset.close()

    def test_one_roll_is_shared_by_every_volume_channel(self) -> None:
        plain = CodecFrameDataset(
            self.catalog,
            family="c5p",
            split="train",
            frames=[0],
            augment=False,
            seed=1701,
        )
        augmented = CodecFrameDataset(
            self.catalog,
            family="c5p",
            split="train",
            frames=[0],
            augment=True,
            seed=1701,
        )
        augmented.set_epoch(7)
        reference = plain[0]["volume"]
        item = augmented[0]
        expected_roll = toroidal_roll(seed=1701, epoch=7, frame=0)

        self.assertEqual(int(item["toroidal_roll"]), expected_roll)
        np.testing.assert_array_equal(
            item["volume"], np.roll(reference, expected_roll, axis=-1)
        )
        plain.close()
        augmented.close()

    def test_epoch_order_is_reproducible_and_complete(self) -> None:
        frames = tuple(range(16))
        first = epoch_order(frames, seed=1701, epoch=3)
        again = epoch_order(frames, seed=1701, epoch=3)
        next_epoch = epoch_order(frames, seed=1701, epoch=4)

        self.assertEqual(first, again)
        self.assertEqual(sorted(first), list(frames))
        self.assertNotEqual(first, next_epoch)

    def test_o2_h2_window_order_and_target_are_exact(self) -> None:
        dataset = OneStepWindowDataset(
            self.catalog,
            split="train",
            target_frames=[2],
            context_frames=2,
            augment=False,
            seed=1701,
            return_physical=True,
        )
        item = dataset[0]

        self.assertEqual(tuple(item["context"].shape), (2, 5, *VOLUME_SHAPE))
        self.assertEqual(tuple(item["target"].shape), (5, *VOLUME_SHAPE))
        np.testing.assert_array_equal(item["context_frame_indices"], [0, 1])
        self.assertEqual(int(item["target_frame_index"]), 2)
        self.assertEqual(dataset.consumed_frames, (0, 1, 2))
        self.assertEqual(float(item["physical_context"][0, 1, 0, 0, 0]), 3.0)
        self.assertEqual(float(item["physical_context"][1, 1, 0, 0, 0]), 13.0)
        self.assertEqual(float(item["physical_target"][1, 0, 0, 0]), 23.0)
        self.assertNotIn("time", item)
        dataset.close()

    def test_o2_roll_is_identical_for_every_context_and_target_field(self) -> None:
        plain = OneStepWindowDataset(
            self.catalog,
            split="train",
            target_frames=[2],
            context_frames=2,
            augment=False,
            seed=1702,
        )
        augmented = OneStepWindowDataset(
            self.catalog,
            split="train",
            target_frames=[2],
            context_frames=2,
            augment=True,
            seed=1702,
        )
        augmented.set_epoch(5)
        reference = plain[0]
        actual = augmented[0]
        expected_roll = toroidal_roll(seed=1702, epoch=5, frame=2)
        self.assertEqual(int(actual["toroidal_roll"]), expected_roll)
        np.testing.assert_array_equal(
            actual["context"], np.roll(reference["context"], expected_roll, axis=-1)
        )
        np.testing.assert_array_equal(
            actual["target"], np.roll(reference["target"], expected_roll, axis=-1)
        )
        plain.close()
        augmented.close()

    def test_o2_validation_window_stays_after_guard(self) -> None:
        dataset = OneStepWindowDataset(
            self.catalog,
            split="validation",
            target_frames=[498],
            context_frames=2,
            augment=False,
            seed=1703,
        )
        item = dataset[0]
        np.testing.assert_array_equal(item["context_frame_indices"], [496, 497])
        self.assertEqual(int(item["target_frame_index"]), 498)
        self.assertEqual(dataset.consumed_frames, (496, 497, 498))
        dataset.close()

    def test_o2_forecast_context_never_reads_or_returns_target_truth(self) -> None:
        dataset = OneStepContextDataset(
            self.catalog,
            target_frames=[498],
            context_frames=2,
            return_physical=True,
        )
        item = dataset[0]
        np.testing.assert_array_equal(item["context_frame_indices"], [496, 497])
        self.assertEqual(int(item["target_frame_index"]), 498)
        self.assertEqual(dataset.consumed_frames, (496, 497))
        self.assertFalse(dataset.target_truth_read)
        self.assertFalse(item["target_truth_read"])
        self.assertNotIn("target", item)
        self.assertNotIn("physical_target", item)
        self.assertEqual(tuple(item["context"].shape), (2, 5, *VOLUME_SHAPE))
        self.assertEqual(
            tuple(item["physical_context"].shape), (2, 5, *VOLUME_SHAPE)
        )
        dataset.close()

    def test_o2_target_guards_reject_unmatched_or_guard_crossing_windows(self) -> None:
        with self.assertRaisesRegex(ValueError, "frozen interval"):
            strict_o2_targets([1], split="train", context_frames=1)
        with self.assertRaisesRegex(ValueError, "frozen interval"):
            strict_o2_targets([496], split="validation", context_frames=2)
        with self.assertRaisesRegex(ValueError, "contiguous"):
            strict_o2_targets([2, 4], split="train", context_frames=2)
        self.assertEqual(
            strict_o2_targets(
                [2, 4],
                split="train",
                context_frames=2,
                allow_sparse_targets=True,
            ),
            (2, 4),
        )
        with self.assertRaisesRegex(ValueError, "one or two"):
            strict_o2_targets([2], split="train", context_frames=3)

    def test_guard_and_validation_augmentation_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "frozen"):
            _strict_contiguous_frames([431, 432], split="train")
        with self.assertRaisesRegex(ValueError, "prohibited"):
            CodecFrameDataset(
                self.catalog,
                family="c5p",
                split="validation",
                frames=[496],
                augment=True,
                seed=1701,
            )

    def test_shard_tampering_is_detected_before_tensor_read(self) -> None:
        shard = self.root / "shards/model_dataset_shard_001.h5"
        with h5py.File(shard, "r+") as handle:
            handle.attrs["tampered"] = True

        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.catalog.verify_consumed_frames([78])

    def test_normalization_round_trip_does_not_clip(self) -> None:
        fields = FAMILY_FIELDS["c5p"]
        raw = [
            np.full(VOLUME_SHAPE, 0.25 + index, dtype=np.float32)
            for index in range(len(fields))
        ]
        encoded = self.catalog.normalization.encode_volume(fields, raw)
        decoded = self.catalog.normalization.decode_volume(fields, encoded)

        np.testing.assert_allclose(decoded, np.stack(raw), rtol=2e-7, atol=2e-7)


if __name__ == "__main__":
    unittest.main()
