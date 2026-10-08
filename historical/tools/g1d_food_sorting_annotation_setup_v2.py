import csv
import json
import os
import re
from pathlib import Path

DATASET = Path("/mnt/data/yangky/test/datasets/g1d_food_sorting_123_lerobot")
ANNOTATION = Path("/mnt/data/yangky/test/xr_teleoperate_g1d/annotation_food_sorting_123")
SOURCE_ANNOTATION = Path("/mnt/data/yangky/test/xr_teleoperate_g1d/G1-d/annotation")
VIDEO_DIR = ANNOTATION / "videos"
MULTIVIEW_DIR = ANNOTATION / "videos_4view"


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
    ingredient_zh = zh_ingredient(ingredient)
    return [
        {
            "object": "move_black_cup",
            "en": "Move the black cup from the right side to the middle position.",
            "zh": "\u5c06\u53f3\u4fa7\u7684\u9ed1\u8272\u676f\u5b50\u79fb\u52a8\u5230\u4e2d\u95f4\u4f4d\u7f6e\u3002",
        },
        {
            "object": "place_ingredient",
            "en": f"Place {ingredient} into the black cup.",
            "zh": f"\u5c06{ingredient_zh}\u653e\u5165\u9ed1\u8272\u676f\u5b50\u3002",
        },
        {
            "object": "left_hand_get_white_cup",
            "en": "Use the left hand to pick up the white cup on the left.",
            "zh": "\u7528\u5de6\u624b\u62ff\u8d77\u5de6\u4fa7\u7684\u767d\u8272\u6c34\u676f\u3002",
        },
        {
            "object": "right_hand_hold_black_cup",
            "en": "Use the right hand to pick up and steady the black cup for pouring.",
            "zh": "\u7528\u53f3\u624b\u63a5\u8fc7\u6216\u7a33\u4f4f\u767d\u8272\u6c34\u676f\uff0c\u51c6\u5907\u5012\u6c34\u3002",
        },
        {
            "object": "pour_water",
            "en": "Pour water from the white cup into the black cup.",
            "zh": "\u5c06\u767d\u8272\u6c34\u676f\u91cc\u7684\u6c34\u5012\u5165\u9ed1\u8272\u676f\u5b50\u3002",
        },
    ]


def patch_index() -> None:
    index = ANNOTATION / "index.html"
    text = (SOURCE_ANNOTATION / "index.html").read_text(encoding="utf-8")
    text = text.replace("G1-D pick_place sub-task labeler", "G1-D food sorting sub-task labeler")
    text = text.replace("G1-D 子任务边界标注", "G1-D food sorting 子任务边界标注")
    text = text.replace(
        "${i === 3 ? 'disabled' : ''}",
        "${i === curLabel.subtasks.length - 1 ? 'disabled' : ''}",
    )
    text = text.replace(
        "<code>1</code>–<code>4</code> 选中子任务 ·",
        "<code>1</code>–<code>5</code> 选中子任务 ·",
    )
    text = text.replace(
        "视频三个视角依次为：左侧头视角(color_0) ｜ 左侧腕部(color_3) ｜ 右侧腕部(color_2)。",
        "视频四个视角依次为：头部主视角 ｜ 左腕 ｜ 右腕 ｜ 备用/重复视角。",
    )
    text = text.replace(
        "子任务4的结束帧固定为最后一帧",
        "最后一个子任务的结束帧固定为最后一帧",
    )
    text = text.replace(
        "所以一条数据只需要标 <b>3 个分割点</b>：子任务 1/2/3 的终点（也就是 2/3/4 的起点）。",
        "所以一条数据只需要标 <b>4 个分割点</b>：子任务 1/2/3/4 的终点（也就是后续子任务的起点）。",
    )
    text = re.sub(
        r"const OBJ_ZH = \{.*?\};\nconst OBJ_TEXT = \{.*?\};",
        """const OBJ_ZH = {
  move_black_cup:'移动黑杯到中间',
  place_ingredient:'放入配料',
  left_hand_get_white_cup:'左手拿白色水杯',
  right_hand_hold_black_cup:'右手拿/稳住黑杯',
  pour_water:'倒水到黑杯'
};
const OBJ_TEXT = {
  move_black_cup: {en:'Move the black cup from the right side to the middle position.', zh:'将右侧的黑色杯子移动到中间位置。'},
  place_ingredient: {en:'Place the ingredient into the black cup.', zh:'将配料放入黑色杯子。'},
  left_hand_get_white_cup: {en:'Use the left hand to pick up the white cup on the left.', zh:'用左手拿起左侧的白色水杯。'},
  right_hand_hold_black_cup: {en:'Use the right hand to pick up and steady the black cup for pouring.', zh:'用右手拿起并稳住黑色杯子，准备倒水。'},
  pour_water: {en:'Pour water from the white cup into the black cup.', zh:'将白色水杯里的水倒入黑色杯子。'},
};""",
        text,
        flags=re.S,
    )
    text = text.replace("Math.max(0, Math.min(3, i))", "Math.max(0, Math.min(curLabel.subtasks.length - 1, i))")
    text = text.replace("else if (k >= '1' && k <= '4') setActive(+k - 1);", "else if (k >= '1' && k <= '9') setActive(+k - 1);")
    index.write_text(text, encoding="utf-8")


def write_instructions() -> None:
    info = json.loads((DATASET / "meta/info.json").read_text(encoding="utf-8"))
    episodes = []
    for line in (DATASET / "meta/episodes.jsonl").read_text(encoding="utf-8").splitlines():
        episode_meta = json.loads(line)
        episode_index = int(episode_meta["episode_index"])
        episode_name = f"episode_{episode_index:06d}"
        episode_id = f"g1d_food_sorting_123_{episode_name}"
        video = MULTIVIEW_DIR / f"{episode_id}.mp4"
        if not video.exists():
            video = DATASET / "videos/chunk-000/observation.images.egocentric" / f"{episode_name}.mp4"
        link = VIDEO_DIR / f"{episode_id}.mp4"
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
        "wording_variant": "food_sorting_123_5subtasks_v2",
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


def main() -> None:
    ANNOTATION.mkdir(parents=True, exist_ok=True)
    (ANNOTATION / "labels").mkdir(exist_ok=True)
    VIDEO_DIR.mkdir(exist_ok=True)
    MULTIVIEW_DIR.mkdir(exist_ok=True)
    (ANNOTATION / "server.py").write_text((SOURCE_ANNOTATION / "server.py").read_text(encoding="utf-8"), encoding="utf-8")
    patch_index()
    write_instructions()
    print("updated food-sorting annotation config")


if __name__ == "__main__":
    main()
