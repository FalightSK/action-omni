from huggingface_hub import HfApi
api = HfApi()

print("=== nvidia GR00T ===")
n = 0
for m in api.list_models(author="nvidia", search="GR00T", limit=30):
    print("  ", m.id, "| dl", m.downloads)
    n += 1
if n == 0:
    print("   (none found under author=nvidia)")

print("=== any GR00T-N1* ===")
for m in api.list_models(search="GR00T-N1", limit=15):
    print("  ", m.id, "| dl", m.downloads)

print("=== candidate LIBERO checkpoints: does the repo carry weights+config? ===")
for r in ["lerobot/pi0_libero_base", "lerobot/pi05_libero_base",
          "openroboto-ai/pi05-libero-pytorch", "pepijn223/pi05_libero_6000",
          "lerobot/smolvla_libero", "HuggingFaceVLA/smolvla_libero",
          "moojink/openvla-7b-oft-finetuned-libero-goal",
          "nvidia/Cosmos-Policy-LIBERO-Predict2-2B"]:
    try:
        info = api.repo_info(r)
        fs = [s.rfilename for s in info.siblings]
        w = [f for f in fs if f.endswith(".safetensors")]
        cfg = [f for f in fs if f.endswith(".json")]
        size = None
        try:
            info2 = api.model_info(r, files_metadata=True)
            size = sum(s.size or 0 for s in info2.siblings) / 1e9
        except Exception:
            pass
        print("  %-46s files=%-4d weights=%-3d cfg=%-3d  ~%s GB"
              % (r, len(fs), len(w), len(cfg),
                 ("%.1f" % size) if size else "?"))
    except Exception as e:
        print("  %-46s ERR %s" % (r, type(e).__name__))
