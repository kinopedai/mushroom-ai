# train_onnx_mushroom.py
import argparse  # コマンドライン引数（設定値）を受け取るため
import os        # パス操作に使う
import random    # 乱数（シャッフルなど）に使う
import numpy as np  # 数値計算（評価時の計算に少し使う）
import torch     # PyTorch本体
import torch.nn as nn  # ニューラルネットの部品（損失関数など）
import torch.optim as optim  # 最適化（重みの更新方法）
from torch.utils.data import DataLoader  # データを小分けで取り出す道具
from torchvision import datasets, transforms, models  # 画像読み込み・変換・モデル
from sklearn.metrics import f1_score  # 評価（マクロF1）

# 学習の設定値をコマンドから受け取れるよう設定
def parse_args():
    p = argparse.ArgumentParser(description="Mushroom classifier: train -> export ONNX")
    p.add_argument("--data-root", type=str, default="data", help="train/val を含む親フォルダ")
    p.add_argument("--img", type=int, default=224, help="画像サイズ（正方形にリサイズ）")
    p.add_argument("--batch", type=int, default=32, help="一度に学習する画像の枚数")
    p.add_argument("--epochs", type=int, default=10, help="学習の回数（大きいほど時間↑ 精度↑）")
    p.add_argument("--lr", type=float, default=3e-4, help="学習率（重みをどれくらい動かすか）")
    p.add_argument("--workers", type=int, default=4, help="データ読み込みの並列数（重い時は0に）")
    p.add_argument("--seed", type=int, default=42, help="乱数のタネ（同じ条件で再現用）")
    p.add_argument("--out-pt", type=str, default="artifacts/best.pt", help="学習済み重みの保存先")
    p.add_argument("--out-onnx", type=str, default="artifacts/mushroom.onnx", help="ONNXの出力先")
    return p.parse_args()

# 結果をできるだけ再現しやすくするための固定設定(ブレを少なくする)
def set_seed(seed: int):
    random.seed(seed) # Python標準の乱数を固定
    np.random.seed(seed) # NumPyの乱数を固定
    torch.manual_seed(seed) # PyTorch（CPU）の乱数を固定
    torch.cuda.manual_seed_all(seed) # PyTorch（GPU）の乱数を固定

# 画像の前処理（画像の揃え方を決める）
def build_transforms(img_size: int):
    # 学習用の変換（少しランダムにして、モデルに“いろんな見え方”を教える）
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)), # 少しズームや切り取りで変化をつける
        transforms.RandomHorizontalFlip(), # 左右反転でデータを増やす
        transforms.ToTensor(), # 画像をテンソルに変換
        transforms.Normalize([0.485, 0.456, 0.406], # 色の平均をそろえる（一般的な値）
                             [0.229, 0.224, 0.225]),
    ])

    # 検証用の変換（ランダムなしで“いつも同じ条件”にする）
    val_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)), # 少し大きめに縮小
        transforms.CenterCrop(img_size), # 真ん中を正方形に切り出す
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    return train_tf, val_tf

# フォルダ→データの取り出し口
def build_dataloaders(root: str, train_tf, val_tf, batch: int, workers: int):
    # フォルダ構成をImageFolderで読み込む（クラス名＝サブフォルダ名）
    train_dir = os.path.join(root, "train") # data/train
    val_dir = os.path.join(root, "val") # data/val

    # ImageFolder：train/クラス名/画像… という定番構成を読み取って、ラベルを自動で付けてくれる
    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds   = datasets.ImageFolder(val_dir,   transform=val_tf)

    # データを小分け（バッチ）で取り出せるようにする（学習時は順番をシャッフル）
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,  num_workers=workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch, shuffle=False, num_workers=workers)

    return train_ds, val_ds, train_loader, val_loader

# 既存の有名モデル（ResNet18）を使う（軽めで速い・精度そこそこ）
def build_model(num_classes: int, device: torch.device):
    # 学習済みのResNet18を呼ぶ
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # 最後の全結合層が受け取る“入力の数”を取得
    in_feat = model.fc.in_features
    # 最後だけ“クラス数用の層”に付け替える
    model.fc = nn.Linear(in_feat, num_classes)

    return model.to(device) # CPU/GPUへ載せる

@torch.no_grad() # “学習の記録”を取らない＝軽くて速い
def evaluate(model, val_loader, device):
    model.eval() # 評価モードON（学習時にしか使わない機能をオフ）
    ys, ps = [], [] # 正解と予測をためる箱
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)     # GPU/CPUにデータを移す
        logits = model(x)                     # 予測のスコアを出す（形：[バッチ, クラス数]）
        pred = logits.argmax(1)               # 一番大きいスコアの番号＝予測ラベル
        ys.extend(y.cpu().numpy().tolist())   # 正解ラベルをリストに追加
        ps.extend(pred.cpu().numpy().tolist())# 予測ラベルをリストに追加
    ys = np.array(ys)
    ps = np.array(ps)
    acc = (ys == ps).mean()                   # 正解率
    mf1 = f1_score(ys, ps, average="macro")   # マクロF1（クラス偏りに強い）
    return acc, mf1

def main():
    # 設定値を読む
    args = parse_args()
    # 乱数固定（再現しやすく）
    set_seed(args.seed)
    # GPUがあれば使う
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 画像のそろえ方を作る
    train_tf, val_tf = build_transforms(args.img)
    # フォルダから画像を読み出せるようにする
    train_ds, val_ds, train_loader, val_loader = build_dataloaders(
        args.data_root, train_tf, val_tf, args.batch, args.workers
    )
    # ククラス（フォルダ）の数を自動で取得
    num_classes = len(train_ds.classes)
    # クラス名を表示して確認（例：['Agaricus', 'Boletus', ...]）
    print(f"Classes: {train_ds.classes}")
    # モデルを用意（既成モデルの最後だけ差し替え）
    model = build_model(num_classes, device)
    # ------------------------------------------------------------
    ### ここから要精査
    # 分類の基本的な損失関数（どれだけ間違えたかの物差し）
    criterion = nn.CrossEntropyLoss()
    # 学習の進め方（AdamW）重みの更新ルール
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)  
    # これまでの最高正解率を保持しておく
    best_acc = 0.0
    # ------------------------------------------------------------

    # 学習ループ
    for epoch in range(1, args.epochs + 1):
        model.train() # 学習モードON
        running_loss = 0.0 # この周の損失の合計

        # バッチ（小分けの画像）ごとにループ
        for x, y in train_loader:
            x, y = x.to(device), y.to(device) # データをデバイスへ
            optimizer.zero_grad()             # 前の勾配（学習のメモ）をいったんゼロに
            logits = model(x)                 # 予測を出す（順伝播）
            loss = criterion(logits, y)       # 間違いの量（損失）を計算
            loss.backward()                   # 間違いを元にモデルに学ばせる（逆伝播）
            optimizer.step()                  # すこしだけ重みを直す（学習）
            running_loss += loss.item() * x.size(0) # 合計して平均用にためる

        # 1周終わったら検証データで成績チェック
        val_acc, val_mf1 = evaluate(model, val_loader, device)
        # 学習データ全体の平均損失を取得
        avg_loss = running_loss / len(train_loader.dataset)

        # 最高記録を更新したら、その重みを保存
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), args.out_pt)
            print(f"  -> Saved best weights to {args.out_pt}")

    # ONNXに変換して保存（バッチの次元は可変にする）
    model.eval() # 評価モードON
    dummy = torch.randn(1, 3, args.img, args.img, device=device) # 入力の“形”だけ教えるダミーを取得（形は：[バッチ=1, 色=3, 高さ=img, 幅=img]）
    torch.onnx.export(
        model,                   # 変換対象のモデル
        dummy,                   # 形を伝えるためのダミー
        args.out_onnx,           # 出力ファイル名
        input_names=["input"],   # 入力名
        output_names=["logits"], # 出力名
        dynamic_axes={           # バッチ数だけ“可変”にする（1でも2でもOKに）
            "input": {0: "batch"},
            "logits": {0: "batch"}
        },
        opset_version=17         # ONNXの規格のバージョン（17は新しめで無難）
    )
    print(f"Exported ONNX -> {args.out_onnx}")

if __name__ == "__main__":
    main()
