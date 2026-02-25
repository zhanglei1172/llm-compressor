# import pandas as pd
import json
import shutil
from pathlib import Path

from tqdm import tqdm

# import datasets

"""
in current code init in hf point to dataset
"""

# parquet_files = list(Path('Anilosan15/Turkish_TTS_Data').glob('*.parquet')) # base dataset on hf


# df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)

# ds = datasets.load_dataset(
#     '/dataset/workspace/zhangl98/dataset/qwen-tts/',
#     split='train',
#     verification_mode='no_checks'
# )
ds = []
metaf_to_audio_dir = {
    "/workspace/zhangl98@xiaopeng.com/code/xmart-quantization-evaluation-omni/seedtts_testset/zh/meta_sub500.lst": "/dataset/workspace/zhangl98/dataset/talker/eval_results/zh/",
    "/workspace/zhangl98@xiaopeng.com/code/xmart-quantization-evaluation-omni/seedtts_testset/en/meta_sub500.lst": "/dataset/workspace/zhangl98/dataset/talker/eval_results/en/",
}
for metaf, audio_dir in metaf_to_audio_dir.items():
    with open(
        metaf,
        "r",
        encoding="utf-8",
    ) as f:
        lines = f.readlines()
    data = [line.strip().split("|") for line in lines]
    for idx, (path, text) in enumerate(data):
        ds.append(
            {
                "text": text,
                "audio": f"{audio_dir}/{path}.wav",
            }
        )

# ✅ SHUFFLE DATASET
# df = df.sample(frac=1, random_state=42).reset_index(drop=True)

output_dir = Path("/dataset/workspace/zhangl98/dataset/talker/processed")
output_dir.mkdir(parents=True, exist_ok=True)
audio_dir = output_dir / "audios"
audio_dir.mkdir(exist_ok=True)
jsonl_path = output_dir / "train.jsonl"

with open(jsonl_path, "w", encoding="utf-8") as f_out:
    for idx, row in tqdm(enumerate(ds)):
        # Lưu audio
        # audio_path = row['audio']  # Assuming the audio is already saved at this path
        audio_path = audio_dir / f"audio_{idx}.wav"
        shutil.copy(row["audio"], audio_path)

        entry = {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a high-quality Text-to-Speech (TTS) model. "
                        "Your task is to convert text into natural, fluent, "
                        "and realistic speech."
                    ),
                },
                {"role": "user", "content": row["text"]},
                {"role": "assistant", "content": "<audio>"},
            ],
            "audios": [str(audio_path)],
            "speaker": "f245",
        }

        f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"processed {len(ds)} samples to {jsonl_path}")
