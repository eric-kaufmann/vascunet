from utils.vessel_iterator import VesselIterator
from pathlib import Path

DATA_DIR = Path(r"C:\Users\Eric Kaufmann\workspace\MA\data\carotid_flow_database")


# Load the vessel data
vessels = VesselIterator(DATA_DIR)

# Print the first 5 vessels
for v_idx, vessel in enumerate(vessels):
    if v_idx == 5:
        break
    print(vessel)

# Print the number of vessels
print(len(vessels))

# Print the first vessel
print(vessels[0])

# Print the last vessel
print(vessels[-1])

    