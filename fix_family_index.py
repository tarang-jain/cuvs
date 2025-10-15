#!/usr/bin/env python3
import re

# Read the file
with open('f.json', 'r') as f:
    lines = f.readlines()

# Process lines from 24708 onwards (24707 in 0-indexed)
for i in range(24707, len(lines)):
    # Check if this line contains family_index
    if '"family_index":' in lines[i]:
        # Extract the current value
        match = re.search(r'"family_index":\s*(\d+)', lines[i])
        if match:
            old_value = int(match.group(1))
            new_value = old_value + 1120
            # Replace the old value with the new value
            lines[i] = re.sub(r'"family_index":\s*\d+', f'"family_index": {new_value}', lines[i])

# Write the modified content back to the file
with open('f.json', 'w') as f:
    f.writelines(lines)

print("Successfully updated family_index values from line 24708 onwards.")
print("Changed values from 0-479 to 1120-1599.")








