import re
import pandas as pd 


def extraire_masses(fichier_path):
    """
    Extract the mass values of each segment from a .bioMod file.
    
    Parameters
    ----------
    fichier_path : str
        Path to the .bioMod file.
    
    Returns
    ----------
    masses : dict
        Dictionary mapping each segment name (str) to its mass (float), 
        as specified in the "mass" line in the segment definition.
    """
    masses = {}
    with open(fichier_path, 'r') as fichier:
        contenu = fichier.read()

    segments = re.findall(r'segment\s+(\w+)(.*?)endsegment', contenu, re.DOTALL)

    for nom_segment, bloc in segments:
        match_masse = re.search(r'mass\s+([\d.]+)', bloc)
        if match_masse:
            masse = float(match_masse.group(1))
            masses[nom_segment] = masse

    return masses


# RAC = "/mnt/c/Users/emmam/biobuddy/examples/applied_examples/"
RAC="../../mnt/c/Users/emmam/Documents/GIT/bioptim/bioptim/examples/getting_started/models/" 

all_data = []

for n in range(1,577):
    # filename = RAC + f"athlete_{n}_deleva.bioMod"
    filename ="examples/getting_started/models/pyomecaman.bioMod"
    masses_dict = extraire_masses(filename)
    total_mass = 0
    for segment, masse in masses_dict.items():
        # print(f"{segment} : {masse} kg")
        total_mass+= masse

print(total_mass)
#     masses_dict["total_mass"] = total_mass
#     row = pd.DataFrame([masses_dict], index=[n])
#     all_data.append(row)

# df_all = pd.concat(all_data)
# df_all.index.name = "athlete"

# print(df_all)

# df_all.to_csv(RAC + "masses.csv")
# print("fichier csv ok")