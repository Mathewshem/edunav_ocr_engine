#python - <<'PY'
import json, os
p="models/labels.json"
with open(p,"r",encoding="utf-8") as f: obj=json.load(f)
print("labels type:", type(obj), "value:", obj)
#PY
