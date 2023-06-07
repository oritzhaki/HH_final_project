
import os
import shutil

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from Modules.reorder_files import run as reorder_files_run
from Modules.generate_fixed_data import run as generate_fixed_data
from Modules.current_to_conductivity import run as current_to_conductivity
from Modules.transformation_df import run as transformation_df
from Modules.image_generator import run as save_plots
from Modules.prepare_data_for_cnn import run as prepare_data_for_cnn
from Modules.prepare_data_for_cnn import get_cell_numbers as get_cell_numbers
from Modules.CNN import CNN_train, get_predicted_cells_path
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
save_plots()
cnn_data = prepare_data_for_cnn()
cell_numbers = get_cell_numbers(cnn_data)
model = CNN_train(cnn_data)
predicted_cells_path = get_predicted_cells_path(model, cnn_data, cell_numbers)

for cell in predicted_cells_path:
    MP_RUN(cell)

for cell in predicted_cells_path:
    result_conclusions(cell)
