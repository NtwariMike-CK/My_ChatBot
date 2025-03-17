import os

DATA_PATH = os.path.abspath("Final chatbot/rag project/docs/")
print(f"📂 Checking directory: {DATA_PATH}")  # Debugging

if not os.path.exists(DATA_PATH):
    print(f"❌ Directory does not exist: {DATA_PATH}")
else:
    print(f"✅ Directory found: {DATA_PATH}")
