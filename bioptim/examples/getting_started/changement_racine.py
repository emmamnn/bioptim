import re
import numpy as np 

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

def add_bar_segments(segments):
    """Add bar segments to the model"""
    bar_segments = [
        {
            "name": "LowerBar",
            "content": [
                "\n",
                "segment LowerBar\n",
                "    rt 0 0 0 xyz 0 0 1.55\n",
                "    mesh 0 0 0 \n",
                "    mesh 2.4 0 0 \n",
                "endsegment \n",
                "\n",
                "    marker LowerBarMarker\n",
                "        parent LowerBar\n",
                "    endmarker\n",
                "\n",
            ]
        },
        {
            "name": "UpperBar",
            "content": [
                "segment UpperBarPosition\n",
                "    rt 0 0 0 xyz 0 1.62 2.35 \n",
                "    mesh 0 0 0\n",
                "    mesh 2.4 0 0\n",
                "endsegment \n",
                "\n",
                "    marker UpperBar\n",
                "        parent UpperBarPosition\n",
                "    endmarker\n",
                "\n",
            ]
        }
    ]

    return bar_segments + segments

def reorder_segment(segments, new_order, new_parents):
    segment_dict = {seg['name']: seg['content'] for seg in segments}
    reordered = []

    for name in new_order:
        if name not in segment_dict:
            print(f"⚠️ Le segment '{name}' n'est pas présent dans le modèle.")
            continue

        lines = segment_dict[name]
        parent_name = new_parents.get(name, None)

        new_lines = []
        parent_updated = False
        for line in lines:
            if line.strip().startswith("parent "):
                indent = re.match(r"^(\s*)", line).group(1)
                if parent_name:
                    new_lines.append(f"{indent}parent {parent_name}\n")
                else:
                    # Garder le parent d'origine
                    new_lines.append(line)
                parent_updated = True
            else:
                new_lines.append(line)

        if not parent_updated:
            if parent_name:
                # Ajouter parent si absent dans le segment original
                for i, line in enumerate(new_lines):
                    if line.strip().startswith("segment "):
                        indent = re.match(r"^(\s*)", new_lines[i + 1]).group(1) if i + 1 < len(new_lines) else "    "
                        new_lines.insert(i + 1, f"{indent}parent {parent_name}\n")
                        break

        reordered.append({'name': name, 'content': new_lines})
        segment_dict.pop(name)

    # Ajouter les segments non mentionnés dans new_order (conserver tels quels)
    for name, lines in segment_dict.items():
        reordered.append({'name': name, 'content': lines})

    return reordered




def write_biomod(filepath, segments, header_lines=None):
    with open(filepath, "w") as f:
        if header_lines:
            f.writelines(header_lines)
            f.write("\n")
        for seg in segments:
            f.writelines(seg["content"])
            f.write("\n")

import math

def split_rot_and_position(segments, targets):
    new_segments = []
    name_to_parent = {}
    old_rt_z = {}  # stocker z des anciens rt

    custom_rot_names = {
        "Forearms": "WristRotation",
        "UpperArms": "ElbowRotation",
        "Thorax": "ShoulderRotation",
    }

    # 1. Mapping nom → parent
    for seg in segments:
        name = seg["name"]
        for line in seg["content"]:
            stripped = line.strip()
            if stripped.startswith("parent "):
                parent_name = stripped.split()[1]
                name_to_parent[name] = parent_name
                break


    # 2. Stocker z initiaux
    for seg in segments:
        name = seg["name"]
        for line in seg["content"]:
            if line.strip().startswith("rt "):
                parts = line.strip().split()
                if len(parts) >= 8:
                    z = float(parts[7])
                    old_rt_z[name] = z
                break

    # 3. Construction des nouveaux segments
    for seg in segments:
        name = seg["name"]
        lines = seg["content"]

        if name not in targets:
            new_segments.append(seg)
            continue

        old_parent = name_to_parent.get(name, "")
        rotation_parent = f"{old_parent}Position" if old_parent else ""

        # noms
        rot_name = custom_rot_names.get(name, f"{name}Rotations")
        pos_name = name if name in ["Pelvis"] else f"{name}Position"

        indent = "    "

        # === ROTATION segment ===
        rot_rt_line = f"{indent}rt 0 0 0 xyz 0 0 0 \n"
        if name == "Hands":
            rot_rt_line = f"{indent}rt 0 0 0 xyz 1.2 0 0 \n"

        rot_seg = {
            "name": rot_name,
            "content": [
                "\n",
                f"segment {rot_name}\n",
                f"{indent}parent {rotation_parent}\n",
                rot_rt_line,
                f"{indent}rotations x\n",
                f"endsegment\n",
                "\n",
            ]
        }

        # === POSITION segment ===
        # Extraire les autres infos
        com_line = ""
        mass_line = ""
        inertia_lines = []
        mesh_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("com "):
                com_line = line
            elif stripped.startswith("mass "):
                mass_line = line
            elif stripped.startswith("inertia"):
                inertia_lines.append(line)
            elif len(inertia_lines) > 0 and len(inertia_lines) < 4:
                inertia_lines.append(line)
            elif stripped.startswith("meshfile") or stripped.startswith("meshrt"):
                mesh_lines.append(line)

        # ==== Nouveau rt ====
        rt_x, rt_y, rt_z = 0.0, 0.0, 0.0
        rx, ry, rz = 0.0, 0.0, 0.0

        if name == "Hands":
            # HandsPosition : z = -COM (z du COM)
            z_com = float(com_line.strip().split()[-1])
            rz = -z_com
        elif name == "Forearms":
            rz = -old_rt_z.get("Hands", 0)
        elif name == "UpperArms":
            rz = -old_rt_z.get("Forearms", 0)
        elif name == "Thorax":
            rz = old_rt_z.get("UpperArms", 0)
            rt_y = "pi"  
        elif name == "Pelvis":
            rz = -old_rt_z.get("Thorax", 0)

        pos_rt_line = f"{indent}rt {rt_x} {rt_y} {rt_z} xyz {rx} {ry} {rz} \n"

        pos_seg = {
            "name": pos_name,
            "content": [
                f"segment {pos_name}\n",
                f"{indent}parent {rot_name}\n",
                pos_rt_line,
                com_line,
                mass_line,
            ] + inertia_lines + mesh_lines + [
                f"endsegment\n",
                "\n",
            ]
        }

        new_segments.append(rot_seg)
        new_segments.append(pos_seg)

    return new_segments






def extract_header(filepath):
    """Get header of thhe file"""
    with open(filepath, "r") as f:
        lines = f.readlines()

    header_lines = []
    for line in lines:
        if line.strip().startswith("segment "):
            break
        header_lines.append(line)
    return header_lines


RAC="../../mnt/c/Users/emmam/Documents/GIT/bioptim/bioptim/examples/getting_started/models/" 




#new_order = ['Hands', 'Forearms', 'UpperArms', 'Thorax', 'Head', 'Pelvis', 'RightThigh', 'RightShank', 'RightFoot', 'LeftThigh', 'LeftShank', 'LeftFoot']
new_order = ['Hands', 'Forearms', 'UpperArms', 'Thorax', 'Head', 'Pelvis', 'Thighs', 'Shanks', 'Feet']
new_parents = {'Hands': 'UpperBar', 'Forearms': 'Hands', 'UpperArms': 'Forearms','Thorax': 'UpperArms', 'Pelvis': 'Thorax', 'Head': 'ThoraxPosition'}
targets = ["Hands", "Forearms", "UpperArms", "Thorax", "Pelvis"]

segments = parse_biomod(RAC + "Athlete_10_armMerged.bioMod")
reordered_segments = reorder_segment(segments, new_order, new_parents)
split_segments = split_rot_and_position(reordered_segments, targets)
segments_with_bar = add_bar_segments(split_segments)




#header = extract_header(RAC + "female1_racine_main_armMerged.bioMod")
header = ['version 4\n', '\n', 'gravity 0 0 -9.81\n']

write_biomod(RAC + "Athlete_10_inverse.bioMod", segments_with_bar, header)
print("écriture ok")
