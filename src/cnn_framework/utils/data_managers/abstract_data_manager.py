import random

from ..dimensions import Dimensions


class AbstractDataManager:
    def __init__(self, data_set_dir):
        self.data_set_dir = data_set_dir

    @staticmethod
    def generate_random_anchor(input_dim, model_dim):
        margin = input_dim.difference(model_dim)

        h_start = random.randrange(margin.height + 1)
        w_start = random.randrange(margin.width + 1)
        if margin.depth is not None:
            d_start = random.randrange(margin.depth + 1)
        else:
            d_start = None
        return Dimensions(h_start, w_start, d_start)

    # Transmitted light images
    def get_bright_field_image_path(self, file):
        return ""

    def get_phase_contrast_image_path(self, file):
        return ""

    def get_dic_image_path(self, file):
        return ""

    # Fluorescence images
    def get_dapi_image_path(self, file):
        return ""

    def get_cellmask_image_path(self, file):
        return ""

    def get_fucci_golgi_image_path(self, file):
        return ""

    def get_fluorescent_image_path(self, file):
        return ""

    def get_malat1_image_path(self, file):
        return ""

    def get_fucci_red_image_path(self, file):
        return ""

    def get_fucci_green_image_path(self, file):
        return ""

    def get_tubulin_image_path(self, file):
        return ""

    def get_MKLP1_image_path(self, file):
        return ""

    # Segmentation images
    def get_nucleus_semantic_image_path(self, file):
        return ""

    def get_nucleus_instance_image_path(self, file):
        return ""

    def get_cell_semantic_image_path(self, file):
        return ""

    def get_cell_instance_image_path(self, file):
        return ""

    def get_cell_topology_image_path(self, file):
        return ""

    def get_nucleus_to_segment_image_path(self, file):
        return self.get_dapi_image_path(file)

    def get_cell_to_segment_image_path(self, file):
        return self.get_cellmask_image_path(file)

    # Merged channels
    def get_microscopy_image_path(self, file):
        return ""

    # Others
    def get_cycle_image_path(self, file):
        return ""

    def get_bounding_boxes_path(self):
        return ""
