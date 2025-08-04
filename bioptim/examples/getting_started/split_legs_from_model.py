import numpy as np

def compute_mass_com_inertia(fused_segment):
    """on considère que les deux segments qui résultent du segment fusioné sont les mêmes"""
    com1 = np.zeros(3)

    fused_mass = fused_segment['mass']
    m1 = fused_mass/2
    
    fused_com = np.array(fused_segment['center_of_mass']).reshape(3)
    com1 = (fused_com*fused_mass)/(2*m1)

    fused_inertia = fused_segment['inertia']
    inertia1 = np.zeros((3, 3))
    a, b, c = com1-fused_com

    inertia1[0, 0] = fused_inertia[0, 0]/2 - m1 * (b ** 2 + c ** 2)
    inertia1[0, 1] = fused_inertia[0, 1]/2 + m1 * (-a * b)
    inertia1[0, 2] = fused_inertia[0, 2]/2 + m1 * (-a * c)
    inertia1[1, 0] = fused_inertia[1, 0]/2 + m1 * (-a * b)
    inertia1[1, 1] = fused_inertia[1, 1]/2 - m1 * (c ** 2 + a ** 2)
    inertia1[1, 2] = fused_inertia[1, 2]/2 + m1 * (-b * c)
    inertia1[2, 0] = fused_inertia[2, 0]/2 + m1 * (-a * c)
    inertia1[2, 1] = fused_inertia[2, 1]/2 + m1 * (-b * c)
    inertia1[2, 2] = fused_inertia[2, 2]/2 - m1 * (a ** 2 + b ** 2)


    return m1, com1, inertia1


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

def extract_meshrt_line(segment):
    for line in segment["content"]:
        if line.strip().startswith("meshrt"):
            return line
    return ""

def build_split_segments(old_segment, parent, mass, com, inertia):
    x_avg = 0 
    x_rot = 0 
    z_avg = extract_z(old_segment)
    meshscale_line = extract_meshscale_line(old_segment)
    meshrt_lineR = extract_meshrt_line(old_segment)
    meshrt_lineL = extract_meshrt_line(old_segment)

    old_segment_name = old_segment['name']
    if old_segment_name == "Thighs":
        x_avg = 0.09

        
    if parent == "Pelvis":
        parentR = parent
        parentL = parent
    else:
        parentR = f"Right{parent}"
        parentL = f"Left{parent}" 
    
    if old_segment_name == "Feet":
        x_rot = -0.35
        nameR = "RightFoot"
        nameL = "LeftFoot"
    else:
        nameR = f"Right{old_segment_name[:-1]}" #on prend pas le s final 
        nameL = f"Left{old_segment_name[:-1]}"
         
    if old_segment_name == "Feet":
        meshnameR = "foot"
        meshnameL = "foot"
    elif old_segment_name == "Thighs":
        meshnameR = "thigh"
        meshnameL = "thigh"
    elif old_segment_name == "Shanks":
        meshnameR = "leg_right"
        meshnameL = "leg_left"
        meshrt_lineR = "\tmeshrt 0 pi/36 0 xyz 0 0 0\n"
        meshrt_lineL = "\tmeshrt 0 -pi/36 0 xyz 0 0 0\n"
    else : 
        meshnameR = nameR.lower()
        meshnameL = nameL.lower()
    


    return ({
        "name": nameR,
        "content": [
            f"segment {nameR}\n",
            f"\tparent {parentR}\n",
            f"\trt {x_rot} 0 0 xyz {x_avg} 0.0 {z_avg}\n",
            f"\tcom {com[0]} {com[1]} {com[2]}\n",
            f"\tmass {mass}\n",
            f"\tinertia\n",
            f"\t\t{inertia[0,0]} {inertia[0,1]} {inertia[0,2]}\n",
            f"\t\t{inertia[1,0]} {inertia[1,1]} {inertia[1,2]}\n",
            f"\t\t{inertia[2,0]} {inertia[2,1]} {inertia[2,2]}\n",
            f"\tmeshfile Model_mesh/{meshnameR}.stl\n",
            meshscale_line,
            meshrt_lineR,
            f"endsegment\n"
        ]
    },
    {
        "name": nameL,
        "content": [
            f"segment {nameL}\n",
            f"\tparent {parentL}\n",
            f"\trt {x_rot} 0 0 xyz {-x_avg} 0.0 {z_avg}\n",
            f"\tcom {com[0]} {com[1]} {com[2]}\n",
            f"\tmass {mass}\n",
            f"\tinertia\n",
            f"\t\t{inertia[0,0]} {inertia[0,1]} {inertia[0,2]}\n",
            f"\t\t{inertia[1,0]} {inertia[1,1]} {inertia[1,2]}\n",
            f"\t\t{inertia[2,0]} {inertia[2,1]} {inertia[2,2]}\n",
            f"\tmeshfile Model_mesh/{meshnameL}.stl\n",
            meshscale_line,
            meshrt_lineL,
            f"endsegment\n"
        ]
    })
    


def add_feet_markers(segments):
    MarkerR = {"name": "MarkerR",
            "content": [
                "\tmarker MarkerR\n",
                "\t\tparent RightFoot\n",
                "\t\tposition 0 0.033 -0.2\n",
                "\tendmarker\n",
                "\n"
            ]
        }
    MarkerL = {"name": "MarkerL",
            "content": [
                "\tmarker MarkerL\n",
                "\t\tparent LeftFoot\n",
                "\t\tposition 0 0.033 -0.2\n",
                "\tendmarker\n",
                "\n"
            ]
        }
    
    segments.append(MarkerR)
    segments.append(MarkerL)
    return segments

# === MAIN ===

RAC = "examples/getting_started/models/new_models/"
# RAC="../../mnt/c/Users/emmam/Documents/GIT/bioptim/bioptim/models/merge/" 

for number in range(8,9):
    # number = 10
    if number < 10:
        number = f"0{number}"
    filename = f"Athlete_{number}_armMerged.bioMod"
    base, ext = filename.rsplit('.', 1)
    new_filename = f"{base}_splitLegs.{ext}"

    segments = parse_biomod(RAC + filename)
    header = extract_header(RAC + filename)

    # Paires à spliter
    segment_pairs = [
        ("Thighs","Pelvis"),
        ("Shanks", "Thigh"),
        ("Feet", "Shank"),
    ]
    # segment_pairs = [
    #     ("UpperArms","Thorax"),
    # ]

    # Extraire tous les segments à fusionner
    name_to_segment = {s["name"]: s for s in segments}

    # Supprimer anciens segments
    for fused_segment, _ in segment_pairs:
        segments = [s for s in segments if s["name"] not in [fused_segment]]

    # Construire et ajouter segments fusionnés
    for old_segment, new_parent in segment_pairs:
        seg_old = name_to_segment[old_segment]
        props_old = extract_segment_props(seg_old)
        mass, com, inertia = compute_mass_com_inertia(props_old)
        right, left = build_split_segments(seg_old, new_parent, mass, com, inertia)
        segments.append(right)
        segments.append(left)

    
    # add_feet_markers(segments)
        
        

    # Écriture fichier
    write_biomod(RAC + new_filename, segments, header)
    print(f"Fichier écrit avec jambes splitées : {new_filename}")