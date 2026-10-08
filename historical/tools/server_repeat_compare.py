from dotenv import load_dotenv
load_dotenv('/mnt/data/yangky/test/Psi0/.env')
from pathlib import Path
import dataclasses, json
import numpy as np
from openpi.training import config, data_loader
from openpi import transforms
from openpi_client.websocket_client_policy import WebsocketClientPolicy

REPORT = Path('/mnt/data/yangky/test/xr_teleoperate_g1d/reports/pi05_official_holdout')
cfg = config.get_config('g1d_subtasks_pi05_official_lora_holdout_v1')
split = json.loads((REPORT / 'split_manifest.json').read_text())
vc = dataclasses.replace(cfg.data, repo_id=split['datasets']['validation']['path'])
dc = vc.create(cfg.assets_dirs, cfg.model); ds = data_loader.create_torch_dataset(dc, 32, cfg.model)
repack = transforms.compose(dc.repack_transforms.inputs)
row = json.loads((Path(vc.repo_id) / 'meta/episodes.jsonl').read_text().splitlines()[0])
idx = row['dataset_from_index'] + int(max(0, row['length'] - 32) * .25)
raw = repack(ds[idx]); gt = np.asarray(raw.pop('actions'))[:, :16]
raw = {k: (v.detach().cpu().numpy() if hasattr(v, 'detach') else np.asarray(v) if hasattr(v, 'numpy') else v) for k, v in raw.items()}
client = WebsocketClientPolicy('127.0.0.1', 9016)
preds = [np.asarray(client.infer(raw)['actions']) for _ in range(20)]
client._ws.close(); preds = np.stack(preds)
result = {'index': int(idx), 'repeats': 20, 'mean_std': float(preds.std(axis=0).mean()), 'per_dim_std': preds.std(axis=0).mean(axis=0).tolist(), 'mean_mae_to_gt': float(np.abs(preds.mean(axis=0) - gt).mean()), 'repeat_mae_to_gt_mean': float(np.abs(preds - gt).mean()), 'gt': gt.tolist(), 'prediction_mean': preds.mean(axis=0).tolist()}
(REPORT / 'server_repeat_metrics.json').write_text(json.dumps(result, indent=2)); print(json.dumps({k:v for k,v in result.items() if k not in ('gt','prediction_mean')}, indent=2))
