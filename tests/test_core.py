import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from evaluate import load_queries, load_split, main
from miner.crops import fixed5_boxes, grid_boxes, random_boxes
from miner.encoder import Encoder
from miner.retrieval import csls, recall_at_k, score


class RetrievalTests(unittest.TestCase):
    def test_crops_keep_aspect_and_stay_inside_the_image(self):
        self.assertEqual(fixed5_boxes(640, 480), [
            (0, 0, 384, 288), (256, 0, 640, 288), (0, 192, 384, 480),
            (256, 192, 640, 480), (128, 96, 512, 384)])
        for x1, y1, x2, y2 in fixed5_boxes(301, 507):
            self.assertEqual((x2 - x1, y2 - y1), (180, 304))
            self.assertTrue(0 <= x1 < x2 <= 301 and 0 <= y1 < y2 <= 507)

    def test_grid_and_random_crop_baselines(self):
        self.assertEqual(grid_boxes(300, 600)[-1], (200, 400, 300, 600))
        self.assertEqual(len(grid_boxes(300, 600)), 9)
        self.assertEqual(random_boxes(300, 600), random_boxes(300, 600))
        for x1, y1, x2, y2 in random_boxes(300, 600):
            self.assertTrue(0 <= x1 < x2 <= 300 and 0 <= y1 < y2 <= 600)
        self.assertEqual(len(fixed5_boxes(300, 600, n=2)), 2)

    def test_crop_blending_changes_rank(self):
        texts = np.eye(2)
        images = np.array([[.8, .6], [.6, .8]])
        crops = np.array([[[0., 1.]], [[1., 0.]]])
        sim = score(texts, images, crops, k=0)
        np.testing.assert_allclose(sim, [[.48, .76], [.76, .48]])
        self.assertEqual(recall_at_k(sim, np.array([1, 0]))['R@1'], 100.)

    def test_chunked_scores_match_direct_formula(self):
        rng = np.random.default_rng(7)
        texts, images, crops = rng.random((7, 3)), rng.random((129, 3)), rng.random((129, 5, 3))
        fused = .6 * (texts @ images.T) + .4 * np.einsum('qd,nrd->qnr', texts, crops).max(2)
        expected = (2 * fused - np.sort(fused, axis=1)[:, -10:].mean(1, keepdims=True)
                    - np.sort(fused, axis=0)[-10:].mean(0, keepdims=True))
        np.testing.assert_allclose(score(texts, images, crops), expected)
        np.testing.assert_allclose(score(texts, images, k=0), texts @ images.T)

    def test_invalid_inputs_raise(self):
        with self.assertRaises(ValueError):
            csls(np.empty((0, 2)))
        with self.assertRaises(ValueError):
            score(np.eye(2), np.eye(2), np.ones((1, 5, 2)))
        with self.assertRaises(ValueError):
            score(np.eye(2), np.eye(2), alpha=2)

    def test_split_flattens_captions_and_reads_images_lazily(self):
        rows = [{'image': 'first', 'captions': ['a', 'b']}, {'image': 'second', 'captions': ['c']}]

        class Split(list):
            def select(self, indices):
                return Split(self[i] for i in indices)

            def __getitem__(self, key):
                if key == 'captions':
                    return [row['captions'] for row in self]
                return list.__getitem__(self, key)

        with patch('datasets.load_dataset', return_value=Split(rows)):
            sources, queries, targets = load_split('repo', 'coco')
            self.assertEqual((queries, targets.tolist()), (['a', 'b', 'c'], [0, 0, 1]))
            self.assertEqual((len(sources), sources[1]), (2, 'second'))
            self.assertEqual(len(load_split('repo', 'coco', limit=1)[1]), 2)

    def test_cli_encodes_all_views_and_reports_metrics(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            Image.new('RGB', (20, 10), 'red').save(root / 'a.png')
            Image.new('RGB', (10, 20), 'green').save(root / 'b.png')
            data = {'images': [{'id': 9, 'file_name': 'a.png'}, {'id': 3, 'file_name': 'b.png'}],
                    'annotations': [{'id': 9, 'caption': 'red'}, {'image_id': 3, 'caption': 'green'}]}
            annotations = root / 'captions.json'
            annotations.write_text(json.dumps(data))
            paths, queries, targets = load_queries(annotations, root, limit=1)
            self.assertEqual((len(paths), queries, targets.tolist()), (1, ['red'], [0]))
            encoder = Encoder.__new__(Encoder)
            encoder.device, encoder.image_size = 'cpu', (16, 16)
            seen = []

            def encode_image(pixels, normalize):
                seen.append(len(pixels))
                return torch.nn.functional.normalize(pixels[:, :2], dim=1)

            encoder.preprocess = lambda im: torch.tensor(im.getpixel((0, 0)), dtype=torch.float32)
            encoder.tokenizer = lambda texts: torch.tensor([[1., 0.] if t == 'red' else [0., 1.] for t in texts])
            encoder.model = SimpleNamespace(encode_image=encode_image, encode_text=lambda x, normalize: x)
            output = io.StringIO()
            argv = ['evaluate.py', '--annotations', str(annotations), '--images-dir', str(root), '--batch-size', '1']
            for strategy, view_count in [('fixed', 6), ('grid', 10), ('random', 6)]:
                seen.clear()
                output = io.StringIO()
                with patch('sys.argv', argv + ['--crops', strategy]), \
                     patch('evaluate.Encoder', return_value=encoder), \
                     contextlib.redirect_stdout(output):
                    main()
                self.assertEqual(seen, [view_count, view_count])
                self.assertEqual(output.getvalue().count('R@1=100.00'), 3)


if __name__ == '__main__':
    unittest.main()
