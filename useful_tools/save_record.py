# %% #--------------------------------------------------------------------------------------#
# -----> Total Importations
from PIL import Image
# %% #--------------------------------------------------------------------------------------#
# -----> Total Functions 

def data2tiff(data, filename=f'data.tif'):
    im = Image.fromarray(data)
    im.save(f"{filename}")
    
    print(f"The data have been saved into {filename}.")
    return None