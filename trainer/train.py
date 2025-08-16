print("trainer: hello world")
# ついでに成果物フォルダがマウントされているか確認用
import os, pathlib
pathlib.Path("/workspace/artifacts/.keep").write_text("ok\n", encoding="utf-8")
print("trainer: wrote /workspace/artifacts/.keep")
