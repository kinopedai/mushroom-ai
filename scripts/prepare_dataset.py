# scripts/prepare_dataset.py
# ------------------------------------------------------------
# 目的：
#   Kaggle等から取得した「クラス別フォルダ構成」の画像データを、
#   学習(train)と検証(val)に分割して、data/{train,val}/<class>/ に配置します。
#
# 特徴：
#   - クラスごとに 8:2（デフォルト）で分割
#   - --limit-per-class で「各クラスから最大N枚だけ抽出」→少量動作確認が簡単
#   - jpg/png/jpeg/webp を対象（必要に応じて拡張可能）
#   - --seed で分割を再現可能（同じ乱数シード）
#
# 使い方（例）：各クラス20枚だけ使う
#   - python3 scripts/prepare_dataset.py --src "_downloads/Mushrooms" --dst data --limit-per-class 20 --split 0.8
#
# ------------------------------------------------------------

import argparse
import random
import shutil
from pathlib import Path
from typing import List

# 対象とする画像の拡張子（必要に応じて追加）
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

def list_images(dir_path: Path) -> List[Path]:
    # 指定フォルダ配下の画像ファイル一覧を返す。
    files = []
    for p in dir_path.glob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p)
    return files

def copy_files(files: List[Path], out_dir: Path):
    # ファイル群を out_dir にコピー（ディレクトリが無ければ作成）
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(f, out_dir / f.name)

def main():
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description="クラス別フォルダの画像を train/val に分割するスクリプト")
    parser.add_argument("--src", required=True, help="入力元の親フォルダ（クラス別フォルダが直下に並んでいる想定）")
    parser.add_argument("--dst", default="data", help="出力先の親フォルダ（default: data）")
    parser.add_argument("--split", type=float, default=0.8, help="train比率（0.8=80%がtrain）")
    parser.add_argument("--limit-per-class", type=int, default=None, help="各クラスの上限枚数（少量だけ抽出したい場合）")
    parser.add_argument("--seed", type=int, default=42, help="乱数シード（分割を再現可能にする）")
    # 定義した引数を読み取る
    args = parser.parse_args()

    # src（入力フォルダ）、dst（出力フォルダ）、その下に train と val フォルダを用意する準備
    src = Path(args.src)
    dst = Path(args.dst)
    train_root = dst / "train"
    val_root = dst / "val"
    if not src.exists():
        raise SystemExit(f"[error] 入力フォルダが見つかりません: {src}")

    # 乱数シード固定（毎回同じ分割結果を得る）
    random.seed(args.seed)

    # クラスを取得（例：_downloads/Mushrooms/Agaricus）
    classes = [d for d in src.iterdir() if d.is_dir()]
    if not classes:
        raise SystemExit(f"[error] クラス別フォルダが見つかりません: {src}/*")

    print(f"[info] classes found: {[c.name for c in classes]}")

    # train と val に振り分けた画像の合計枚数を初期化
    total_train = total_val = 0
    # 各クラスフォルダごとにループ
    for c in classes:
        images = list_images(c)
        if not images:
            print(f"[warn] 画像が見つかりません: {c}")
            continue

        # シャッフルしてランダムに分割
        random.shuffle(images)

        # 上限が指定されていたら、各クラスから最大N枚だけ使用
        if args.limit_per_class is not None:
            images = images[:args.limit_per_class]

        # 分割境界（例：100枚なら80枚がtrain、20枚がval）
        k = int(len(images) * args.split)
        train_imgs = images[:k]
        val_imgs = images[k:]

        # コピー実行
        copy_files(train_imgs, train_root / c.name)
        copy_files(val_imgs, val_root / c.name)

        total_train += len(train_imgs)
        total_val += len(val_imgs)
        print(f"[ok] {c.name}: train={len(train_imgs)}, val={len(val_imgs)}")

    print(f"[done] output -> {train_root} (total {total_train})")
    print(f"[done] output -> {val_root}   (total {total_val})")

if __name__ == "__main__":
    main()
