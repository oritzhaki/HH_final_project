
import os
import shutil

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from Modules.gather_data import gather_paths, get_best_cells
from Modules.reorder_files import run as reorder_files_run
from Modules.generate_fixed_data import run as generate_fixed_data
from Modules.current_to_conductivity import run as current_to_conductivity
from Modules.transformation_df import run as transformation_df
from Modules.image_generator import run as save_plots
from Modules.prepare_data_for_cnn import run as prepare_data_for_cnn
from Modules.prepare_data_for_cnn import get_cell_numbers as get_cell_numbers
from Modules.CNN import CNN_train, CNN_predict
from Modules.Algo.MP_RUNNER import run as MP_RUN
from Modules.result_conclusions import run as result_conclusions

dirs_to_check = ["DataOrder", "ConductivityData"]

for directory in dirs_to_check:
    if os.path.exists(directory):
        shutil.rmtree(directory)
        print(f"Deleted existing directory {directory}")

reorder_files_run()
generate_fixed_data()
current_to_conductivity()
transformation_df()
# save_plots()
paths = gather_paths()
cnn_data_train = prepare_data_for_cnn()
model = CNN_train(cnn_data_train)

for p in paths:
    best_cells = get_best_cells(p)
    
    for cell in best_cells:
        
        type_, path = CNN_predict(model, cell, p)
        
        if(type_ == "N"):
            MP_RUN(path)
            result_conclusions(path)
            
        elif(type_ == "MH"):
            print("YET TO IMPLEMENT")
            #MP_RUN(MP_path)
            #result_conclusions(cell)
            