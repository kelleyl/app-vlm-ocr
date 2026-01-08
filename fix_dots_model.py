import json
import os
import glob
from huggingface_hub import snapshot_download

# 1. Download/Locate the model
MODEL_ID = "rednote-hilab/dots.ocr"
print(f"🔍 Locating {MODEL_ID} in cache...")
try:
    model_path = snapshot_download(repo_id=MODEL_ID)
    print(f"   Found at: {model_path}")
except Exception as e:
    print(f"   Error: Could not download model. {e}")
    exit(1)

# 2. Patch preprocessor_config.json
# The app requires 'video_processor_type' to be explicitly defined.
config_file = os.path.join(model_path, "preprocessor_config.json")
if not os.path.exists(config_file):
    config_file = os.path.join(model_path, "config.json")

print(f"🛠  Patching config: {os.path.basename(config_file)}...")

with open(config_file, "r") as f:
    config_data = json.load(f)

if "video_processor_type" not in config_data:
    # We map it to Qwen2VLProcessor which is compatible with the app's structure
    config_data["video_processor_type"] = "Qwen2VLProcessor"
    
    with open(config_file, "w") as f:
        json.dump(config_data, f, indent=2)
    print("   ✅ Added 'video_processor_type': 'Qwen2VLProcessor'")
else:
    print("   ℹ️  'video_processor_type' already exists. Skipping.")

# 3. Patch the Python Code (processing_dots_ocr.py)
# This fixes the crash where the code doesn't accept the 'video_processor' argument
print("🛠  Checking Python code for compatibility...")

py_files = glob.glob(os.path.join(model_path, "*.py"))
target_file = None

for file in py_files:
    with open(file, "r") as f:
        content = f.read()
        if "class DotsOCRProcessor" in content or "class DotsVLProcessor" in content:
            target_file = file
            break

if target_file:
    print(f"   Found processor code: {os.path.basename(target_file)}")
    with open(target_file, "r") as f:
        lines = f.readlines()
    
    new_lines = []
    patched = False
    for line in lines:
        # Inject video_processor=None into __init__ arguments if not present
        if "def __init__" in line and "video_processor" not in line and "self" in line:
            # Handle different common argument orders
            if "tokenizer=None," in line:
                new_line = line.replace("tokenizer=None,", "tokenizer=None, video_processor=None,")
            elif "image_processor=None," in line:
                new_line = line.replace("image_processor=None,", "image_processor=None, video_processor=None,")
            else:
                new_line = line.replace("self,", "self, video_processor=None,")
            new_lines.append(new_line)
            patched = True
        else:
            new_lines.append(line)

    if patched:
        with open(target_file, "w") as f:
            f.writelines(new_lines)
        print("   ✅ Code patched to accept video_processor argument.")
    else:
        print("   ℹ️  Code already patched.")
else:
    print("   ⚠️  Warning: Could not find processor python file.")

print("\n🎉 Patch complete.")