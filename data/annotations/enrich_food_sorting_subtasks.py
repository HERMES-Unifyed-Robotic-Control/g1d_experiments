import csv
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
LABEL_DIR = BASE / "labels"
INSTRUCTIONS = BASE / "instructions.json"

HOME_EN = ", then return the robot to its home pose."
HOME_ZH = "\uff0c\u5b8c\u6210\u540e\u673a\u5668\u4eba\u59ff\u6001\u56de\u6b63\u3002"


def main() -> int:
    meta = {
        episode["id"]: episode
        for episode in json.loads(INSTRUCTIONS.read_text(encoding="utf-8"))["episodes"]
    }
    episodes = []
    for path in sorted(LABEL_DIR.glob("*.json")):
        label = json.loads(path.read_text(encoding="utf-8"))
        episode_meta = meta[path.stem]
        subtasks = []
        label_subtasks = label.get("subtasks", [])
        for index, subtask in enumerate(label_subtasks):
            is_last = index == len(label_subtasks) - 1
            text_en = subtask.get("text_en") or subtask.get("en") or ""
            text_zh = subtask.get("text_zh") or subtask.get("zh") or ""
            if is_last:
                text_en = text_en.rstrip(".") + HOME_EN
                text_zh = text_zh.rstrip("\u3002") + HOME_ZH
            start = subtask.get("start_frame")
            end = subtask.get("end_frame")
            subtasks.append(
                {
                    "index": index,
                    "object": subtask.get("object"),
                    "start_frame": start,
                    "end_frame": end,
                    "num_frames": None if start is None or end is None else end - start,
                    "is_last": is_last,
                    "variant": 0,
                    "text_en": text_en,
                    "text_zh": text_zh,
                }
            )
        episodes.append(
            {
                "id": path.stem,
                "dataset": episode_meta["dataset"],
                "episode": episode_meta["episode"],
                "num_frames": episode_meta["num_frames"],
                "fps": episode_meta["fps"],
                "instruction_en": episode_meta.get("instruction_en", ""),
                "instruction_zh": episode_meta.get("instruction_zh", ""),
                "grasp_order": episode_meta.get("grasp_order", []),
                "subtasks": subtasks,
            }
        )

    out = {
        "num_episodes": len(episodes),
        "num_subtasks": sum(len(episode["subtasks"]) for episode in episodes),
        "num_variants": 1,
        "home_pose_clause": {
            "en": HOME_EN.strip(", ."),
            "zh": HOME_ZH.strip("\uff0c\u3002"),
        },
        "episodes": episodes,
    }
    (BASE / "subtask_labels_enriched.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with (BASE / "subtask_labels_enriched.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "id",
                "dataset",
                "episode",
                "episode_frames",
                "subtask_index",
                "object",
                "start_frame",
                "end_frame",
                "num_frames",
                "is_last",
                "variant",
                "text_en",
                "text_zh",
            ]
        )
        for episode in episodes:
            for subtask in episode["subtasks"]:
                writer.writerow(
                    [
                        episode["id"],
                        episode["dataset"],
                        episode["episode"],
                        episode["num_frames"],
                        subtask["index"] + 1,
                        subtask["object"],
                        subtask["start_frame"],
                        subtask["end_frame"],
                        subtask["num_frames"],
                        int(subtask["is_last"]),
                        subtask["variant"] + 1,
                        subtask["text_en"],
                        subtask["text_zh"],
                    ]
                )

    print(f"episodes={out['num_episodes']} subtasks={out['num_subtasks']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
