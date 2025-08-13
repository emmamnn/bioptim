import re
import numpy as np 
import pandas as pd

 
def parse_biomod(filepath):
    """ 
    Extracts segments from a .bioMod file and stores them in a list of dictionaries
     
    Parameters
    ----------
    filepath : str
        Path to the .bioMod file to parse
    
    Returns
    ----------
    segments : list of dict
        List of dictionaries representing the segments in the file. Each dictionary contains:
            - "name" : name of the segment (str)
            - "content" : list of lines belonging to the segment (list of str)
    """ 
    with open(filepath, "r") as f:
        lines = f.readlines()

    segments = []
    current_segment = None
    inside_segment = False

    for line in lines:
        stripped = line.strip()
        parts = stripped.split()

        if parts and parts[0] == "segment":  
            inside_segment = True
            current_segment = {
                "name": parts[1] if len(parts) > 1 else None,
                "content": [line]
            }
        elif inside_segment and parts and parts[0] == "endsegment":
            current_segment["content"].append(line)
            segments.append(current_segment)
            current_segment = None
            inside_segment = False
        elif inside_segment:
            current_segment["content"].append(line)

    return segments


def add_bar_segments(segments):
    """
    Add bar segments to the list of segments
    
    Parameters
    ----------
    segments : list of dict
        List of existing segments, each represented as a dictionary 
        with keys "name" and "content"
    
    Returns
    ----------
    segments : list of dict
        A new list containing the bar segments ("LowerBar" and "UpperBar") 
        added before the original segments
    """
    bar_segments = [
        {
            "name": "LowerBar",
            "content": [
                "\n",
                "segment LowerBar\n",
                "\trt 0 0 0 xyz 0 0 1.55\n",
                "\tmesh 0 0 0 \n",
                "\tmesh 0 2.4 0 \n",
                "endsegment \n",
                "\n",
                "\tmarker LowerBarMarker\n",
                "\t\tparent LowerBar\n",
                "\tendmarker\n",
                "\n",
            ]
        },
        {
            "name": "UpperBar",
            "content": [
                "segment UpperBarPosition\n",
                "\trt 0 0 0 xyz 1.62 0 2.35 \n",
                "\tmesh 0 0 0\n",
                "\tmesh 0 2.4 0\n",
                "endsegment \n",
                "\n",
                "\tmarker UpperBar\n",
                "\t\tparent UpperBarPosition\n",
                "\tendmarker\n",
                "\n",
            ]
        }
    ]

    return bar_segments + segments
    

def add_last_marker(segments, foot_length):
    """
    Add the feet markers to the list of segments
    
    Parameters
    ----------
    segments : list of dict
        List of existing segments, each represented as a dictionary 
        with keys "name" and "content"
        
    foot_length : float
        Length of the foot, used to set the position of the new markers along the X-axis
    
    Returns
    ----------
    segments : list of dict
        A new list containing the original segments followed by two new markers:
        "MarkerR" (attached to the right foot) and "MarkerL" (attached to the left foot)
    """
    MarkerR = [{"name": "MarkerR",
            "content": [
                "\tmarker MarkerR\n",
                "\t\tparent R_FOOT\n",
                f"\t\tposition {foot_length} 0 0\n",
                "\tendmarker\n",
                "\n"
            ]
        } ]
    MarkerL = [{"name": "MarkerL",
            "content": [
                "\tmarker MarkerL\n",
                "\t\tparent L_FOOT\n",
                f"\t\tposition {foot_length} 0 0\n",
                "\tendmarker\n",
                "\n"
            ]
        }]

    return segments + MarkerR + MarkerL

def modify_hands_and_remove_root(segments):
    """
    Modify the "HANDS" segment and remove the "root" segment from the list of segments
    
    Parameters
    ----------
    segments : list of dict
        List of existing segments, each represented as a dictionary 
        with keys "name" and "content"
    
    Returns
    ----------
    updated_segments : list of dict
        A new list of segments where:
        - The segment named "root" is removed
        - In the segment named "HANDS":
            * The parent is changed to "UpperBarPosition"
            * The RT matrix is replaced with an identity matrix
            * The segment is positioned in the middle of the bar along the Y-axis
        - All other segments remain unchanged
    """
    
    updated_segments = []

    for seg in segments:
        # remove root segment 
        if seg["name"] == "root":
            continue

        if seg["name"] == "HANDS":
            new_content = []
            for line in seg["content"]:
                stripped = line.strip()

                # change the parent
                if stripped.startswith("parent"):
                    new_content.append("\tparent\tUpperBarPosition\n")
                    continue
                
                # change the RT matrix 
                if stripped == "RT":
                    new_content.append("\tRT\n")
                    new_content.append("\t\t1.000000\t0.000000\t0.000000\t0.000000\n")
                    new_content.append("\t\t0.000000\t1.000000\t0.000000\t1.200000\n")
                    new_content.append("\t\t0.000000\t0.000000\t1.000000\t0.000000\n")
                    new_content.append("\t\t0.000000\t0.000000\t0.000000\t1.000000\n")
                    continue

                # skip old RT lines to remove them from the content of the segment  
                if re.match(r"^[\t\s]*-?\d+\.\d+\t-?\d+\.\d+\t-?\d+\.\d+\t-?\d+\.\d+", line):
                    continue

                new_content.append(line)

            updated_segments.append({"name": seg["name"], "content": new_content})
        else:
            updated_segments.append(seg)

    return updated_segments


def write_biomod(filepath, segments, header_lines=None):
    """
    Write the list of segments in a .bioMod file
    
    Parameters
    ----------
    filepath : str
        Path to the output .bioMod file.
    
    segments : list of dict
        List of existing segments, each represented as a dictionary 
        with keys "name" and "content"
    
    
    header_lines : list of str, optional
        Optional list of lines to write at the beginning of the file before the segments.
    """
    
    with open(filepath, "w") as f:
        if header_lines:
            f.writelines(header_lines)
            f.write("\n")
        for seg in segments:
            f.writelines(seg["content"])
            f.write("\n")
        f.close()



def update_dof_and_rangesQ(segments, dof_and_ranges):
    """
    Update the degrees of freedom (DoF) and rangesQ of segments
    
    Parameters
    ----------
    segments : list of dict
        List of existing segments, each represented as a dictionary 
        with keys "name" and "content"
    
    dof_and_ranges : dict
        Dictionary mapping segment names to new DoF and ranges information.
        Each value is a dictionary that may contain:
            - "translations" : list of str, new translation DoF
            - "rotations" : list of str, new rotation DoF
            - "rangesQ" : list of tuple(float, float), new min/max ranges for the DoF
    
    Returns
    ----------
    updated_segments : list of dict
        A new list of segments where:
        - Existing translations, rotations, and rangesQ lines are removed for segments 
          not present in `dof_and_ranges`.
        - For segments present in `dof_and_ranges`, new translations, rotations, and 
          rangesQ lines are inserted before the "mass" line.
    """
          
    updated_segments = []

    for seg in segments:
        name = seg["name"]
        new_lines = []
        i = 0

        # Remove existing DOF lines 
        while i < len(seg["content"]):
            line = seg["content"][i]
            stripped = line.strip()

            if stripped.startswith(("translations", "rotations", "rangesQ")):
                # Skip all rangesQ values if rangesQ line
                i += 1
                while i < len(seg["content"]) and re.match(r"\s*-?\d+\.?\d*\s+-?\d+\.?\d*", seg["content"][i].strip()):
                    i += 1
                continue
            new_lines.append(line)
            i += 1

        # Add the new DOF if the segment is in dof_and_ranges
        if name in dof_and_ranges:
            info = dof_and_ranges[name]
            ranges_iter = iter(info.get("rangesQ", []))

            # find where to insert the DOF lines (before the "mass" line)
            insert_index = next(
                (i for i, line in enumerate(new_lines) if line.strip().startswith("mass")), len(new_lines)
            )
            indent = re.match(r"^(\s*)", new_lines[insert_index] if insert_index < len(new_lines) else "").group(1)

            insertion = []
            if info.get("translations"):
                insertion.append(f"{indent}translations {' '.join(info['translations'])}\n")
            if info.get("rotations"):
                insertion.append(f"{indent}rotations {' '.join(info['rotations'])}\n")

            # Add rangesQ line if given 
            ranges_list = info.get("rangesQ", [])
            if ranges_list:
                insertion.append(f"{indent}rangesQ\n")
                for min_val, max_val in ranges_list:
                    insertion.append(f"{indent}    {min_val} {max_val}\n")

            new_lines = new_lines[:insert_index] + insertion + new_lines[insert_index:]

        updated_segments.append({
            "name": name,
            "content": new_lines
        })

    return updated_segments



def rotate_upper_trunk(segments):
    """
    Apply a rotation of pi radians around the Y-axis to the "UPPER_TRUNK" segment
    in order to place the gymnast in handstand position 
    
    Parameters
    ----------
    segments : list of dict
        List of model segments, each represented as a dictionary 
        with keys "name" and "content".
    
    Returns
    ----------
    updated_segments : list of dict
        A new list of segments where:
        - The "UPPER_TRUNK" segment's RT matrix is replaced with a rotated version:
            [-1, 0, 0, 0]
            [0, 1, 0, 0]
            [0, 0, -1, original_z_translation]
            [0, 0, 0, 1]
        - All other segments remain unchanged.
    """
    updated_segments = []

    for seg in segments:
        if seg["name"] == "UPPER_TRUNK":
            new_content = []
            inside_rt = False
            rt_line_index = 0
            translation_z = None

            for line in seg["content"]:
                stripped = line.strip()

                if stripped.startswith("RTinMatrix"):
                    new_content.append(line)
                    continue

                if stripped.startswith("RT"):
                    new_content.append(line) 
                    inside_rt = True
                    rt_line_index = 0
                    continue

                if inside_rt:
                    rt_line_index += 1
                    if rt_line_index == 3:  
                        parts = line.split()
                        if len(parts) >= 4:
                            translation_z = float(parts[3])
                        
                        new_content.append(f"\t\t0.000000\t0.000000\t-1.000000\t{translation_z:.6f}\n")
                        continue
                    elif rt_line_index == 1:  
                        new_content.append("\t\t-1.000000\t0.000000\t0.000000\t0.000000\n")
                        continue
                    elif rt_line_index == 2:  
                        new_content.append("\t\t0.000000\t1.000000\t0.000000\t0.000000\n")
                        continue
                    elif rt_line_index == 4:  
                        inside_rt = False
                        new_content.append("\t\t0.000000\t0.000000\t0.000000\t1.000000\n")
                        continue

                new_content.append(line)

            updated_segments.append({"name": seg["name"], "content": new_content})
        else:
            updated_segments.append(seg)

    return updated_segments


dof_and_rom = {
    "HANDS": {
        "translations" : ["xz"],
        "rotations": ["y"],
        "rangesQ": [(-0.1, 0.1), (-0.1, 0.1), ("-2*pi", "2*pi")]
    },
    "UPPER_ARMS": {
        "rotations": ["y"],
        "rangesQ": [("-2*pi/9", 0)] #40°
    },
    "UPPER_TRUNK": {
        "rotations": ["y"],
        "rangesQ": [("-2*pi/9", 0)] #40°
    },
    "HEAD": {
        "rotations": ["y"],
        "rangesQ": [("-pi/3", "5*pi/18")]  #-60° , 50°
    },
    "LOWER_TRUNK": {
        "rotations": ["y"],
        "rangesQ": [("-pi/18", "5*pi/36")] #-10°, 25° 
    },
    "R_THIGH": {
        "rotations": ["xy"],
        "rangesQ": [("-pi/3", 0), ("-pi/3", "pi/6")] #-60° 0°  | -60°, 30° 
    },
     "L_THIGH": {
        "rotations": ["xy"],
        "rangesQ": [(0, "pi/3"), ("-pi/3", "pi/6")] #0° 60° | -60°, 30°
    },
    "R_SHANK": {
        "rotations": ["y"],
        "rangesQ": [(0, "5*pi/6")]
    },
     "L_SHANK": {
        "rotations": ["y"],
        "rangesQ": [(0, "5*pi/6")]
    },
    "L_FOOT": {
        "rotations": ["y"],
        "rangesQ": [("-7*pi/18", 0)] 
    },
    "R_FOOT": {
        "rotations": ["y"],
        "rangesQ": [("-7*pi/18", 0)] 
    },
}

RAC = "/mnt/c/Users/emmam/biobuddy/examples/applied_examples/"
data = RAC + "model_coefficients.csv" 
df = pd.read_csv(data, usecols=["ModelNumber", "Height", "FootLengthCoeff"], sep=";")
# FootLengthCoeff = df["FootLengthCoeff"]
# Height = df["Height"]
df["FootLength"] = df["FootLengthCoeff"] * df["Height"]



for num in range(1,len(df.index)+1):
    filename = f"athlete_{num}_deleva.bioMod"
    segments = parse_biomod(RAC + filename)
    segments = rotate_upper_trunk(segments)
    segments_with_bar = add_bar_segments(segments)
    segments_bis = modify_hands_and_remove_root(segments_with_bar)
    segments_with_marker = add_last_marker(segments_bis,df["FootLength"][num-1])
    all_segments = update_dof_and_rangesQ(segments_with_marker, dof_and_rom)

    header = ['version 4\n', '\n', 'gravity 0 0 -9.81\n']
    write_biomod(RAC + filename, all_segments, header)
    print("écriture ok : ", filename)



