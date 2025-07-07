import numpy as np

def compute_mass_com_inertia(segments, xyz_global=np.zeros(3)):
    total_mass = 0
    com = np.zeros(3)

    for seg in segments:
        mass = seg['mass']
        com_local = np.array(seg['center_of_mass']).reshape(3)
        com += mass * (com_local - xyz_global)
        total_mass += mass

    if total_mass == 0:
        raise ValueError("La masse totale est nulle.")

    com /= total_mass

    inertia = np.zeros((3, 3))
    for seg in segments:
        current_com = np.array(seg['center_of_mass']) - xyz_global
        dist = current_com - com
        a, b, c = dist
        rel_inertia = seg['inertia']
        mass = seg['mass']

        inertia[0, 0] += rel_inertia[0, 0] + mass * (b ** 2 + c ** 2)
        inertia[0, 1] += rel_inertia[0, 1] - mass * (-a * b)
        inertia[0, 2] += rel_inertia[0, 2] - mass * (-a * c)
        inertia[1, 0] += rel_inertia[1, 0] - mass * (-a * b)
        inertia[1, 1] += rel_inertia[1, 1] + mass * (c ** 2 + a ** 2)
        inertia[1, 2] += rel_inertia[1, 2] - mass * (-b * c)
        inertia[2, 0] += rel_inertia[2, 0] - mass * (-a * c)
        inertia[2, 1] += rel_inertia[2, 1] - mass * (-b * c)
        inertia[2, 2] += rel_inertia[2, 2] + mass * (a ** 2 + b ** 2)

    return total_mass, com, inertia

def parse_biomod(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()

    segments = []
    current_segment = None
    inside_segment = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("segment "):
            inside_segment = True
            current_segment = {
                "name": stripped.split()[1],
                "content": [line]
            }
        elif inside_segment and stripped.startswith("endsegment"):
            current_segment["content"].append(line)
            segments.append(current_segment)
            current_segment = None
            inside_segment = False
        elif inside_segment:
            current_segment["content"].append(line)

    return segments

def write_biomod(filepath, segments, header_lines=None):
    with open(filepath, "w") as f:
        if header_lines:
            f.writelines(header_lines)
            f.write("\n")
        for seg in segments:
            f.writelines(seg["content"])
            f.write("\n")

def extract_header(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()

    header_lines = []
    for line in lines:
        if line.strip().startswith("segment "):
            break
        header_lines.append(line)
    return header_lines

def extract_segment_props(segment):
    mass = 0
    com = np.zeros(3)
    inertia = np.zeros((3, 3))
    for line in segment["content"]:
        if line.strip().startswith("mass"):
            mass = float(line.strip().split()[1])
        elif line.strip().startswith("com"):
            com = np.array([float(x) for x in line.strip().split()[1:]])
        elif line.strip().startswith("inertia"):
            idx = segment["content"].index(line)
            inertia = np.array([
                [float(x) for x in segment["content"][idx+1].strip().split()],
                [float(x) for x in segment["content"][idx+2].strip().split()],
                [float(x) for x in segment["content"][idx+3].strip().split()]
            ])
    return {
        "mass": mass,
        "center_of_mass": com,
        "inertia": inertia
    }

def extract_z(segment):
    for line in segment["content"]:
        if "rt" in line and "xyz" in line:
            parts = line.strip().split()
            if "xyz" in parts:
                idx = parts.index("xyz")
                try:
                    return float(parts[idx + 3])
                except (IndexError, ValueError):
                    return 0.0
    return 0.0

def extract_meshscale_line(segment):
    for line in segment["content"]:
        if line.strip().startswith("meshscale"):
            return line
    return "\tmeshscale 1 1 1\n"

def build_fused_segment(name, parent, mass, com, inertia, segment_left, segment_right):
    z_avg = (extract_z(segment_left) + extract_z(segment_right)) / 2
    meshscale_line = extract_meshscale_line(segment_left)

    return {
        "name": name,
        "content": [
            f"segment {name}\n",
            f"\tparent {parent}\n",
            f"\trt 0 0 0 xyz 0.0 0.0 {z_avg}\n",
            f"\tcom {com[0]} {com[1]} {com[2]}\n",
            f"\tmass {mass}\n",
            f"\tinertia\n",
            f"\t\t{inertia[0,0]} {inertia[0,1]} {inertia[0,2]}\n",
            f"\t\t{inertia[1,0]} {inertia[1,1]} {inertia[1,2]}\n",
            f"\t\t{inertia[2,0]} {inertia[2,1]} {inertia[2,2]}\n",
            f"\tmeshfile Model_mesh/{name.lower()}.stl\n",
            meshscale_line,
            f"endsegment\n"
        ]
    }

# === MAIN ===

RAC = "../../mnt/c/Users/emmam/Documents/GIT/bioptim/bioptim/examples/getting_started/models/"
filename = "Athlete_10.bioMod"
base, ext = filename.rsplit('.', 1)
new_filename = f"{base}_armMerged.{ext}"

segments = parse_biomod(RAC + filename)
header = extract_header(RAC + filename)

# Paires à combiner
segment_pairs = [
    ("LeftUpperArm", "RightUpperArm", "UpperArms", "Thorax"),
    ("LeftForearm", "RightForearm", "Forearms", "UpperArms"),
    ("LeftHand", "RightHand", "Hands", "Forearms"),
]

# Extraire tous les segments à fusionner
name_to_segment = {s["name"]: s for s in segments}

# Supprimer anciens segments
for left, right, _, _ in segment_pairs:
    segments = [s for s in segments if s["name"] not in [left, right]]

# Construire et ajouter segments fusionnés
for left, right, new_name, parent in segment_pairs:
    seg_left = name_to_segment[left]
    seg_right = name_to_segment[right]
    props_left = extract_segment_props(seg_left)
    props_right = extract_segment_props(seg_right)
    mass, com, inertia = compute_mass_com_inertia([props_left, props_right])
    fused = build_fused_segment(new_name, parent, mass, com, inertia, seg_left, seg_right)
    segments.append(fused)

# Écriture fichier
write_biomod(RAC + new_filename, segments, header)
print(f"Fichier écrit avec bras, avant-bras et mains combinés : {new_filename}")
