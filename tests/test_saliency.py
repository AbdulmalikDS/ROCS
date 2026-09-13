import sys
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch

import torch


@unittest.skipUnless(find_spec('cv2'), 'Install appendix/saliency/requirements.txt')
class DINOCheckpointTests(unittest.TestCase):
    def test_local_checkpoint_is_loaded_without_a_download(self):
        from appendix.saliency import dinov3_saliency

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoint = root / 'weights.pth'
            torch.save({'weight': torch.tensor([[3.]])}, checkpoint)
            (root / 'hubconf.py').write_text('import unused_segmentation_dependency\n')
            package = root / 'dinov3' / 'hub'
            package.mkdir(parents=True)
            (root / 'dinov3' / '__init__.py').touch()
            (package / '__init__.py').touch()
            (package / 'backbones.py').write_text('''
import torch

def dinov3_vitl16(*, pretrained, weights):
    model = torch.nn.Linear(1, 1, bias=False)
    model.load_state_dict(torch.load(weights, weights_only=True))
    return model
''')
            with patch.object(dinov3_saliency, '_DINOV3_REPO', root), \
                 patch('sys.path', sys.path[:]), patch.dict(sys.modules):
                model = dinov3_saliency._load_dinov3(device='cpu', weights=checkpoint)
                self.assertEqual(model(torch.tensor([[2.]])).item(), 6.)
                self.assertFalse(model.training)
                with self.assertRaises(FileNotFoundError):
                    dinov3_saliency._load_dinov3(device='cpu', weights=root / 'missing.pth')
                with self.assertRaisesRegex(ValueError, 'checkpoint'):
                    dinov3_saliency._load_dinov3(device='cpu')


if __name__ == '__main__':
    unittest.main()
