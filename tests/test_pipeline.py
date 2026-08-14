import tempfile
import unittest
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from PIL import Image

from checkpoint import build_checkpoint, load_checkpoint
from dataset import ImageSegmentationDataset
from model import ARCHITECTURE_NAME, UNetResNet
from preprocessing import IMAGENET_MEAN, IMAGENET_STD, image_transform
from train_model import CombinedSegmentationLoss, spatial_split


class PipelineTests(unittest.TestCase):
    @staticmethod
    def write_pair(root: Path, stem: str, foreground_columns: int = 3) -> None:
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        image[:, :foreground_columns] = 255
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[:, :foreground_columns] = 255
        Image.fromarray(image).save(root / "images" / f"{stem}.png")
        Image.fromarray(mask).save(root / "masks" / f"{stem}.png")

    def test_model_output_and_synthetic_training_step(self):
        model = UNetResNet(pretrained=False)
        model.train()
        inputs = torch.randn(1, 3, 64, 64)
        targets = torch.randint(0, 2, (1, 64, 64))
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        optimizer.zero_grad(set_to_none=True)
        output = model(inputs)

        self.assertEqual(tuple(output.shape), (1, 2, 64, 64))
        loss = CombinedSegmentationLoss(torch.ones(2))(output, targets)
        loss.backward()
        optimizer.step()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.decoder1.refine[0].weight.grad)
        self.assertIsNotNone(model.stem[0].weight.grad)

    def test_checkpoint_round_trip_and_rejects_legacy_state_dict(self):
        model = UNetResNet(pretrained=False)
        checkpoint = build_checkpoint(model, 3, {"mean_iou": 0.75}, threshold=0.65)

        self.assertEqual(checkpoint["architecture"], ARCHITECTURE_NAME)
        self.assertEqual(tuple(checkpoint["normalization"]["mean"]), IMAGENET_MEAN)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "model.pth"
            torch.save(checkpoint, checkpoint_path)
            restored, metadata = load_checkpoint(checkpoint_path, torch.device("cpu"))

            self.assertEqual(metadata["inference_threshold"], 0.65)
            self.assertEqual(
                tuple(restored(torch.zeros(1, 3, 64, 64)).shape),
                (1, 2, 64, 64),
            )

            legacy_path = Path(directory) / "legacy.pth"
            torch.save(model.state_dict(), legacy_path)
            with self.assertRaisesRegex(ValueError, "legado ou incompatível"):
                load_checkpoint(legacy_path, torch.device("cpu"))

    def test_dataset_pairs_by_stem_and_rejects_missing_masks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images").mkdir()
            (root / "masks").mkdir()
            self.write_pair(root, "bloco_1_0", foreground_columns=2)
            self.write_pair(root, "bloco_0_0", foreground_columns=4)

            dataset = ImageSegmentationDataset(root / "images", root / "masks")
            paired_stems = [
                (image_path.stem, mask_path.stem)
                for image_path, mask_path in dataset.samples
            ]
            self.assertEqual(
                paired_stems,
                [("bloco_0_0", "bloco_0_0"), ("bloco_1_0", "bloco_1_0")],
            )

            Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(
                root / "images" / "bloco_2_0.png"
            )
            with self.assertRaisesRegex(ValueError, "sem máscara"):
                ImageSegmentationDataset(root / "images", root / "masks")

    def test_joint_transform_keeps_binary_mask_aligned(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images").mkdir()
            (root / "masks").mkdir()
            self.write_pair(root, "bloco_0_0")

            dataset = ImageSegmentationDataset(
                root / "images",
                root / "masks",
                joint_transform=A.Compose(
                    [
                        A.Resize(16, 16),
                        A.HorizontalFlip(p=1.0),
                    ]
                ),
            )
            transformed_image, transformed_mask = dataset[0]
            image_foreground = transformed_image.mean(dim=0) > 0.5

            self.assertEqual(transformed_mask.dtype, torch.long)
            self.assertEqual(set(transformed_mask.unique().tolist()), {0, 1})
            self.assertTrue(torch.equal(image_foreground, transformed_mask.bool()))

    def test_normalization_matches_imagenet_constants(self):
        image = Image.fromarray(np.full((2, 2, 3), 255, dtype=np.uint8))
        tensor = image_transform()(image)
        expected = torch.tensor(
            [(1 - mean) / std for mean, std in zip(IMAGENET_MEAN, IMAGENET_STD)]
        )
        self.assertTrue(torch.allclose(tensor[:, 0, 0], expected))

    def test_spatial_split_has_no_column_leakage(self):
        samples = [
            (Path(f"bloco_{x}_{y}.png"), Path(f"bloco_{x}_{y}.png"))
            for x in range(5)
            for y in range(3)
        ]
        training, validation = spatial_split(samples, 0.2)
        training_columns = {
            int(samples[index][0].stem.split("_")[1]) for index in training
        }
        validation_columns = {
            int(samples[index][0].stem.split("_")[1]) for index in validation
        }

        self.assertFalse(training_columns & validation_columns)
        self.assertEqual(validation_columns, {4})


if __name__ == "__main__":
    unittest.main()
