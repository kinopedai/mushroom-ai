DC := docker-compose

# ----------------------
# コンテナ ビルド
# ----------------------
# docker compose build
# 各サービス（trainer / infer）のDockerイメージをビルド
build:
	$(DC) build

# ----------------------
# trainer 1回だけ実行
# ----------------------
train:
	$(DC) run --rm trainer

# ----------------------
# infer バックグラウンドで起動
# ----------------------
up:
	$(DC) up -d infer

# ----------------------
# infer の疎通確認
# ----------------------
# curl: infer の API にアクセスして正常応答を確認
# /health → サーバが動作中かチェック
# /hello  → 実際のエンドポイント動作確認（Hello Worldなど）
test:
	@echo "GET /health"
	@curl -s http://localhost:8080/health ; echo
	@echo "GET /hello"
	@curl -s http://localhost:8080/hello ; echo

# ----------------------
# コンテナ 停止＆削除
# ----------------------
# docker compose down
down:
	$(DC) down
