from __future__ import annotations

try:
    from lerobot.common.datasets.lerobot_dataset import (  # type: ignore
        LeRobotDataset,
        LeRobotDatasetMetadata,
        MultiLeRobotDataset,
    )

    LEROBOT_LAYOUT = "common"
except ModuleNotFoundError:
    from lerobot.datasets.lerobot_dataset import (  # type: ignore
        LeRobotDataset,
        LeRobotDatasetMetadata,
        MultiLeRobotDataset,
    )

    LEROBOT_LAYOUT = "datasets"

__all__ = [
    "LEROBOT_LAYOUT",
    "LeRobotDataset",
    "LeRobotDatasetMetadata",
    "MultiLeRobotDataset",
]
