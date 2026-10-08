import csv
import json
import os
import re
from pathlib import Path

DATASET = Path("/mnt/data/yangky/test/datasets/g1d_food_sorting_123_lerobot")
ANNOTATION = Path("/mnt/data/yangky/test/xr_teleoperate_g1d/annotation_food_sorting_123")
SOURCE_ANNOTATION = Path("/mnt/data/yangky/test/xr_teleoperate_g1d/G1-d/annotation")


def ingredient_from_instruction(text: str) -> str:
    match = re.search(r"place (.*?) into it", text)
    return match.group(1) if match else "ingredient"


def zh_ingredient(ingredient: str) -> str:
    return {
        "chrysanthemum": "\u83ca\u82b1",
        "red dates": "\u7ea2\u67a3",
        "lemon slices": "\u67e0\u6aac\u7247",
    }.get(ingredient, ingredient)


def subtask_templates(instruction: str) -> list[dict]:
    ingredient = ingredient_from_instruction(instruction)
    return [
        {
            "object": "black_cup",
            "en": "Move the black cup from the right side to the middle position.",
            "zh": "\u5c06\u53f3\u4fa7\u7684\u9ed1\u8272\u676f\u5b50\u79fb\u52a8\u5230\u4e2d\u95f4\u4f4d\u7f6e\u3002",
        },
        {
            "object": ingredient.replace(" ", "_"),
            "en": f"Place {ingredient} into the black cup.",
            "zh": f"\u5c06{zh_ingredient(ingredient)}\u653e\u5165\u9ed1\u8272\u676f\u5b50\u3002",
        },
        {
            "object": "white_cup_water",
            "en": "Pour water from the white cup on the left into the black cup.",
            "zh": "\u5c06\u5de6\u4fa7\u767d\u8272\u676f\u5b50\u91cc\u7684\u6c34\u5012\u5165\u9ed1\u8272\u676f\u5b50\u3002",
        },
    ]


def main() -> None:
    ANNOTATION.mkdir(parents=True, exist_ok=True)
    (ANNOTATION / "labels").mkdir(exist_ok=True)
    (ANNOTATION / "videos").mkdir(exist_ok=True)

    for name in ("server.py", "index.html"):
        target = ANNOTATION / name
        target.write_text((SOURCE_ANNOTATION / name).read_text(encoding="utf-8"), encoding="utf-8")

    index = ANNOTATION / "index.html"
    text = index.read_text(encoding="utf-8")
    text = text.replace("${i === 3 ? 'disabled' : ''}", "${i === curLabel.subtasks.length - 1 ? 'disabled' : ''}")
    text = text.replace("G1-D pick_place sub-task labeler", "G1-D food sorting sub-task labeler")
    text = text.replace("G1-D pick_place \u5b50\u4efb\u52a1\u8fb9\u754c\u6807\u6ce8", "G1-D food sorting \u5b50\u4efb\u52a1\u8fb9\u754c\u6807\u6ce8")
    index.write_text(text, encoding="utf-8")

    info = json.loads((DATASET / "meta/info.json").read_text(encoding="utf-8"))
    episodes = []
    for line in (DATASET / "meta/episodes.jsonl").read_text(encoding="utf-8").splitlines():
        episode_meta = json.loads(line)
        episode_index = int(episode_meta["episode_index"])
        episode_name = f"episode_{episode_index:06d}"
        episode_id = f"g1d_food_sorting_123_{episode_name}"
        video = DATASET / "videos/chunk-000/observation.images.egocentric" / f"{episode_name}.mp4"
        link = ANNOTATION / "videos" / f"{episode_id}.mp4"
        if link.exists() or link.is_symlink():
            link.unlink()
        os.symlink(video, link)

        subtasks = subtask_templates(episode_meta.get("instruction", ""))
        episodes.append(
            {
                "id": episode_id,
                "dataset": "g1d_food_sorting_123_lerobot",
                "episode": episode_name,
                "path": str(DATASET / "data/chunk-000" / f"{episode_name}.parquet"),
                "video_path": str(video),
                "num_frames": int(episode_meta["length"]),
                "fps": int(info.get("fps", 30)),
                "task_index": episode_meta.get("task_index"),
                "instruction_en": episode_meta.get("instruction", ""),
                "instruction_zh": "",
                "grasp_order": [subtask["object"] for subtask in subtasks],
                "subtasks": subtasks,
            }
        )

    payload = {
        "objects": {},
        "wording_variant": "food_sorting_123_v1",
        "num_episodes": len(episodes),
        "episodes": episodes,
    }
    (ANNOTATION / "instructions.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with (ANNOTATION / "instructions.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "episode", "num_frames", "task_index", "instruction_en", "subtasks"])
        for episode in episodes:
            writer.writerow(
                [
                    episode["id"],
                    episode["episode"],
                    episode["num_frames"],
                    episode["task_index"],
                    episode["instruction_en"],
                    " > ".join(episode["grasp_order"]),
                ]
            )

    print(f"wrote {ANNOTATION / 'instructions.json'} episodes={len(episodes)}")
    print(f"linked videos={len(list((ANNOTATION / 'videos').glob('*.mp4')))}")


if __name__ == "__main__":
    main()
