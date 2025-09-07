# mushroom-ai

AI推論システムのDocker環境。トレーニング用Pythonコンテナと推論用Goサーバで構成。

## 構成

- **trainer**: Python環境でのモデルトレーニング
- **infer**: Go製HTTP推論サーバー (ポート8080)

## 使用方法

```bash
# ビルド
make build

# トレーニング実行
make train

# 推論サーバー起動
make up

# API疎通確認
make test

# 停止
make down
```

## API エンドポイント

- `GET /health` - ヘルスチェック
- `GET /hello` - Hello World応答