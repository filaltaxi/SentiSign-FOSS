# run_pipeline.py
# ─────────────────────────────────────────────────────────────────────────────
# SentiSign — Main Pipeline Entry Point  [Phase 2]
#
# Phase 1 (done): Manual words + emotion input
# Phase 2 (done): Emotion auto-detected from webcam via ResNet CNN
# Phase 3 (next): Words auto-recognised from webcam via Sign Recognition
#
# Pipeline:
#   words (still manual)  +  emotion (ResNet webcam detection)
#       │                          │
#       │               continuous webcam → emotion counts
#       │               → winner or user resolves tie
#       └──────────┬───────────────┘
#                  ▼  sentence model
#       grammatically correct English sentence
#                  ▼  Chatterbox-TTS (exaggeration + cfg_weight)
#              🔊  emotion-expressive audio
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
if ".venv" not in sys.executable and "venv" not in sys.executable:
    print("⚠  WARNING: venv does not appear to be active.")
    print("   Run: .venv\\Scripts\\activate.bat")
    print("   Then re-run the pipeline.\n")

_ROOT    = os.path.dirname(os.path.abspath(__file__))
_SRC     = os.path.join(_ROOT, "src")
_SLM_SRC = os.path.join(_ROOT, "slm", "src")
OUTPUT_DIR = _ROOT

for p in [_SRC, _SLM_SRC]:
    if p not in sys.path:
        sys.path.insert(0, p)

from emotion_map       import print_emotion_table
from generate_sentence import words_to_sentence
from sentence_model    import get_backend_label
from tts               import get_output_extension, speak_and_save
from sign_recognizer   import capture_words_and_emotion


# ── Input parsing ─────────────────────────────────────────────────────────────


def get_inputs() -> tuple[list[str], str]:
    print("\n" + "═" * 64)
    print("  SentiSign  |  Sign Language → Emotion-Aware Speech")
    print(f"  Models: {get_backend_label()}  +  Chatterbox-TTS  +  ResNet Emotion")
    print("═" * 64)
    print_emotion_table()

    # ── Step 1: Words (still manual — Phase 3 will automate) ─────────────────
    print("\n  Next: webcam opens ? sign your words AND show your emotion together.")
    print("  GREEN box = hand recognition   BLUE box = face emotion")
    input("  Press ENTER here to open the webcam > ")
    words, emotion = capture_words_and_emotion()
    return words, emotion


# ── Pipeline run ──────────────────────────────────────────────────────────────

def run():
    words, emotion = get_inputs()

    # Step 1 — sentence generation
    print("\n" + "─" * 64)
    print("  [1/2]  Generating sentence from words ...")
    sentence = words_to_sentence(words)
    print(f"\n  ✓  Sentence: \"{sentence}\"")

    # Step 2 — TTS with emotion
    print("\n" + "─" * 64)
    print(f"  [2/2]  Synthesising speech  (emotion: {emotion}, engine: Chatterbox) ...")

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ext = get_output_extension()
    fname = f"sentisign_{emotion}_{timestamp}.{ext}"
    path  = os.path.join(OUTPUT_DIR, fname)
    print(f"\n  Auto-saving audio as: {fname}")
    speak_and_save(sentence, emotion=emotion, path=path, also_play=True)
    print(f"\n  ✓  Saved: {path}")

    # Summary
    print("\n" + "═" * 64)
    print("  ✓  Pipeline complete")
    print(f"  Words    : {words}")
    print(f"  Sentence : \"{sentence}\"")
    print(f"  Emotion  : {emotion}")
    print("  Engine   : Chatterbox")
    print("═" * 64)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n  All models load on the FIRST run only.")
    print("  Subsequent runs reuse loaded models — much faster.\n")

    while True:
        try:
            run()
        except KeyboardInterrupt:
            print("\n\n  Stopped.")
            break
        except Exception as e:
            print(f"\n  ✗  Error: {e}")
            import traceback; traceback.print_exc()

        again = input("\n  Run again? (y/n) > ").strip().lower()
        if again != "y":
            print("  Goodbye.")
            break
