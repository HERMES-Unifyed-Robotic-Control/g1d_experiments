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
dc = vc.create(cfg.assets_dirs, cfg.model)
ds = data_loader.create_torch_dataset(dc, 32, cfg.model)
repack = transforms.compose(dc.repack_transforms.inputs)
client = WebsocketClientPolicy('127.0.0.1', 9016)
print('SERVER_METADATA', json.dumps(client.get_server_metadata()), flush=True)
preds, targets, tasks = [], [], []
rows = [json.loads(line) for line in (Path(vc.repo_id) / 'meta/episodes.jsonl').read_text().splitlines()]
for row in rows:
    usable = max(0, row['length'] - 32)
    for frac in (.25, .75):
        idx = row['dataset_from_index'] + int(usable * frac)
        raw = repack(ds[idx])
        gt = np.asarray(raw.pop('actions'))[:, :16]
        raw = {k: (v.detach().cpu().numpy() if hasattr(v, 'detach') else np.asarray(v) if hasattr(v, 'numpy') else v) for k, v in raw.items()}
        out = np.asarray(client.infer(raw)['actions'])
        assert out.shape == (32, 16) and np.isfinite(out).all()
        preds.append(out); targets.append(gt); tasks.append(row['task_index'])
        if len(preds) % 10 == 0:
            print(f'INFERRED {len(preds)}/200', flush=True)
client._ws.close()
preds, targets, tasks = np.stack(preds), np.stack(targets), np.asarray(tasks)
err = preds - targets
result = {
    'observations': len(preds),
    'overall': {
        'mae': float(np.abs(err).mean()),
        'rmse': float(np.sqrt((err * err).mean())),
        'left_arm_mae': float(np.abs(err[:, :, :7]).mean()),
        'right_arm_mae': float(np.abs(err[:, :, 7:14]).mean()),
        'left_gripper_mae': float(np.abs(err[:, :, 14]).mean()),
        'right_gripper_mae': float(np.abs(err[:, :, 15]).mean()),
    },
    'source': '9016 WebsocketPolicyServer, same 200 heldout indices as heldout_action_metrics.json',
    'note': 'Server infer uses internal sampling; no robot commands.',
}
(REPORT / 'server_holdout_metrics.json').write_text(json.dumps(result, indent=2))
np.savez_compressed(REPORT / 'server_holdout_predictions.npz', predictions=preds, targets=targets, task_indices=tasks)
print(json.dumps(result, indent=2), flush=True)
