# clean_control_chars.py
import re, sys

if len(sys.argv) != 2:
    print("Usage: python clean_control_chars.py train_diffusion.py")
    sys.exit(1)

path = sys.argv[1]
# Read the file as UTF-8 (ignore any bad sequences)
text = open(path, "r", encoding="utf-8", errors="ignore").read()

# Remove any control character in U+0000–U+001F except clean whitespace (\n, \r, \t)
cleaned = re.sub(r"[\x00-\x08\x0B-\x1F]", "", text)

# Overwrite the file as UTF-8
with open(path, "w", encoding="utf-8") as f:
    f.write(cleaned)

print(f"Cleaned control characters from {path}")