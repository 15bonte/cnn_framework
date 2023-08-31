import glob
import os
import random

from .dimensions import Dimensions


def read_type_from_image_path_function(function):
    str_fct = function.__name__
    start, end = "get_", "_image_path"
    return str_fct[str_fct.find(start) + len(start) : str_fct.rfind(end)]


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


class DefaultDataManager(AbstractDataManager):
    def get_distinct_files(self):
        return os.listdir(self.data_set_dir)

    def get_microscopy_image_path(self, file):
        return os.path.join(self.data_set_dir, file)


class MaxenceData1Manager(AbstractDataManager):
    # ch1 = Contrast ; ch2 = Cy3 ; ch3 = GFP ; ch4 = DAPI ; ch5 = BrightField
    def get_distinct_files(self):
        files = os.listdir(self.data_set_dir)
        return list({file.split("_c")[0] for file in files})

    def get_dapi_image_path(self, file):
        return os.path.join(self.data_set_dir, f"{file}_ch4.tiff")

    def get_bright_field_image_path(self, file):
        return os.path.join(self.data_set_dir, f"{file}_ch5.tiff")

    def get_phase_contrast_image_path(self, file):
        return os.path.join(self.data_set_dir, f"{file}_ch1.tiff")

    def get_fucci_golgi_image_path(self, file):
        return os.path.join(self.data_set_dir, f"{file}_ch3.tiff")

    def get_nucleus_semantic_image_path(self, file):
        # Should be updated
        return os.path.join(self.data_set_dir, file + "_cellpose.png")

    def get_nucleus_instance_image_path(self, file):
        return os.path.join(self.data_set_dir, file + "_cnucleus_instance.png")


class MaxenceData2Manager(AbstractDataManager):
    def get_distinct_files(self):
        files = os.listdir(self.data_set_dir)
        return list(
            {file.split("_")[0] + " " + file.split("_")[-1].split(".")[0] for file in files}
        )

    def get_dapi_image_path(self, file):
        splitted_file = file.split(" ")
        return os.path.join(
            self.data_set_dir, f"{splitted_file[0]}_w3Hoechst_{splitted_file[1]}.TIF"
        )

    def get_cellmask_image_path(self, file):
        splitted_file = file.split(" ")
        return os.path.join(self.data_set_dir, f"{splitted_file[0]}_w4Cy5_{splitted_file[1]}.TIF")

    def get_dic_image_path(self, file):
        splitted_file = file.split(" ")
        return os.path.join(
            self.data_set_dir, f"{splitted_file[0]}_w5DIC-oil-40x_{splitted_file[1]}.TIF"
        )

    def get_cell_semantic_image_path(self, file):
        splitted_file = file.split(" ")
        return os.path.join(
            self.data_set_dir, f"{splitted_file[0]}_cell_semantic_{splitted_file[1]}.png"
        )

    def get_cell_topology_image_path(self, file):
        splitted_file = file.split(" ")
        return os.path.join(
            self.data_set_dir, f"{splitted_file[0]}_cell_topology_{splitted_file[1]}.png"
        )

    def get_cell_instance_image_path(self, file):
        splitted_file = file.split(" ")
        return os.path.join(
            self.data_set_dir, f"{splitted_file[0]}_cell_instance_{splitted_file[1]}.png"
        )

    def get_nucleus_instance_image_path(self, file):
        splitted_file = file.split(" ")
        return os.path.join(
            self.data_set_dir, f"{splitted_file[0]}_nucleus_instance_{splitted_file[1]}.png"
        )

    def get_nucleus_semantic_image_path(self, file):
        splitted_file = file.split(" ")
        return os.path.join(
            self.data_set_dir, f"{splitted_file[0]}_nucleus_semantic_{splitted_file[1]}.png"
        )

    @staticmethod
    def generate_random_anchor(input_dim, model_dim):
        margin = input_dim.difference(model_dim)

        # Ignore top zone as light is bad
        minimum_height = int(input_dim.height / 3)
        if minimum_height > margin.height:
            raise ValueError(
                f"Not possible to crop with height {margin.height} if top {minimum_height}px are forbidden."
            )

        h_start = random.randrange(minimum_height, margin.height)
        w_start = random.randrange(margin.width)
        if margin.depth is not None:
            d_start = random.randrange(margin.depth)
        else:
            d_start = None
        return Dimensions(h_start, w_start, d_start)


class ChristiansenDataManager(AbstractDataManager):
    def get_distinct_files(self):
        files = os.listdir(self.data_set_dir)
        # Careful, specific to Rubin\scott_1_0
        return list({",".join(file.split(",")[:7]) for file in files})  # distinct

    def get_bright_field_image_path(self, file, idx):
        return os.path.join(
            self.data_set_dir,
            file
            + f",z_depth-{idx},channel,value-BRIGHTFIELD,is_mask-false,kind,value-ORIGINAL.png",
        )

    def get_fluorescent_image_path(self, file):
        return os.path.join(
            self.data_set_dir,
            file + ",depth_computation,value-MAXPROJECT,is_mask-false,kind,value-ORIGINAL.png",
        )


class OunkomolDataManager(AbstractDataManager):
    def get_distinct_files(self):
        return os.listdir(self.data_set_dir)


class EmelineFucciDataManager(AbstractDataManager):
    def get_distinct_files(self):
        files = os.listdir(self.data_set_dir)
        return list({"_".join(file.split("_")[:2]) for file in files})

    def get_malat1_image_path(self, file):
        return os.path.join(self.data_set_dir, f"{file}_MALAT1.tiff")

    def get_fucci_red_image_path(self, file):
        return os.path.join(self.data_set_dir, f"{file}_Fucci_red.tiff")

    def get_fucci_green_image_path(self, file):
        return os.path.join(self.data_set_dir, f"{file}_Fucci_green.tiff")

    def get_dapi_image_path(self, file):
        return os.path.join(self.data_set_dir, f"{file}_DAPI.tiff")

    def get_dic_image_path(self, file):
        return os.path.join(self.data_set_dir, f"{file}_dic.tiff")

    def get_nucleus_instance_image_path(self, file):
        return os.path.join(self.data_set_dir, f"{file}_nucleus_instance.png")

    def get_cycle_image_path(self, file):
        return os.path.join(self.data_set_dir, f"{file}_cycle.tiff")


class PasteurDataManager(AbstractDataManager):
    def get_distinct_files(self):
        files = glob.glob(os.path.join(self.data_set_dir, "*.tif"))
        return list({os.path.basename(file).split("_ch")[0] for file in files})

    def get_tubulin_image_path(self, file):
        return os.path.join(self.data_set_dir, f"{file}_ch1.tif")

    def get_MKLP1_image_path(self, file):
        return os.path.join(self.data_set_dir, f"{file}_ch2.tif")

    def get_phase_contrast_image_path(self, file):
        return os.path.join(self.data_set_dir, f"{file}_ch3.tif")

    def get_bounding_boxes_path(self):
        # return one (should be unique) json file in folder
        return list(glob.glob(os.path.join(self.data_set_dir, "*.json")))[0]

    def get_cell_to_segment_image_path(self, file):
        return self.get_phase_contrast_image_path(file)

    def get_cell_instance_image_path(self, file):
        return os.path.join(self.data_set_dir, f"{file}_cell_instance.png")


class BridgesDataManager(AbstractDataManager):
    def get_distinct_files(self):
        files = glob.glob(os.path.join(self.data_set_dir, "*.tif"))
        return list({os.path.basename(file).split(".")[0] for file in files})

    def get_microscopy_image_path(self, file):
        return os.path.join(self.data_set_dir, f"{file}.tif")

    def get_bounding_boxes_path(self):
        # return one (should be unique) json file in folder
        return list(glob.glob(os.path.join(self.data_set_dir, "*.json")))[0]


class MarieCecileData1Manager(AbstractDataManager):
    def get_distinct_files(self):
        files = os.listdir(self.data_set_dir)
        return list({file.split("-")[0] + "-" + file.split("-")[1] for file in files})

    def get_malat1_image_path(self, file):
        raise ValueError("Cell segmentation is not available in this dataset")

    def get_fucci_red_image_path(self, file):
        return os.path.join(self.data_set_dir, f"{file}-Cy3-sk1fk1fl1.tiff")

    def get_fucci_green_image_path(self, file):
        return os.path.join(self.data_set_dir, f"{file}-EGFP-sk1fk1fl1.tiff")

    def get_dapi_image_path(self, file):
        return os.path.join(self.data_set_dir, f"{file}-DAPI-sk1fk1fl1.tiff")

    def get_bright_field_image_path(self, file):
        return os.path.join(self.data_set_dir, f"{file}-Brightfield-sk1fk1fl1.tiff")

    def get_nucleus_instance_image_path(self, file):
        return os.path.join(self.data_set_dir, f"{file}-nucleus_instance.png")

    def get_cycle_image_path(self, file):
        return os.path.join(self.data_set_dir, f"{file}-cycle.tiff")


class MarieCecileData2Manager(AbstractDataManager):
    def get_distinct_files(self):
        files = os.listdir(self.data_set_dir)
        return list({file.split("-")[0] for file in files})

    def get_malat1_image_path(self, file):
        return os.path.join(self.data_set_dir, f"{file}-ch5sk1fk1fl1.tiff")

    def get_fucci_red_image_path(self, file):
        return os.path.join(self.data_set_dir, f"{file}-ch1sk1fk1fl1.tiff")

    def get_fucci_green_image_path(self, file):
        return os.path.join(self.data_set_dir, f"{file}-ch2sk1fk1fl1.tiff")

    def get_dapi_image_path(self, file):
        return os.path.join(self.data_set_dir, f"{file}-ch3sk1fk1fl1.tiff")

    def get_bright_field_image_path(self, file):
        return os.path.join(self.data_set_dir, f"{file}-ch4sk1fk1fl1.tiff")

    def get_nucleus_instance_image_path(self, file):
        return os.path.join(self.data_set_dir, f"{file}-nucleus_instance.png")

    def get_cycle_image_path(self, file):
        return os.path.join(self.data_set_dir, f"{file}-cycle.tiff")


class FloricDataManager(AbstractDataManager):
    @staticmethod
    def get_prefix(file):
        splitted_file = file.split("-")
        splitted_prefix = []
        for s in splitted_file:
            splitted_prefix.append(s)
            if len(s) > 2 and s[-3] == "f":  # fov indicator
                break
        return "-".join(splitted_prefix)

    def get_distinct_files(self):
        files = os.listdir(self.data_set_dir)
        return list({self.get_prefix(file) for file in files})

    def get_dapi_image_path(self, file):
        return os.path.join(self.data_set_dir, f"{file}-DAPI-sk1fk1fl1.tiff")

    def get_nucleus_instance_image_path(self, file):
        return os.path.join(self.data_set_dir, f"{file}-nucleus_instance.png")


class NathalieDataManager(AbstractDataManager):
    def get_distinct_files(self):
        files = os.listdir(self.data_set_dir)
        return list({"_".join(file.split("_")[:-1]) for file in files})

    def get_phase_contrast_image_path(self, file):
        return os.path.join(self.data_set_dir, f"{file}_TRANS.tif")

    def get_tubulin_image_path(self, file):
        return os.path.join(self.data_set_dir, f"{file}_Cy5.tif")

    def get_MKLP1_image_path(self, file):
        return os.path.join(self.data_set_dir, f"{file}_FITC.tif")


class OrianeDataManager(AbstractDataManager):
    def get_distinct_files(self):
        files = os.listdir(self.data_set_dir)
        return list({file.split("-")[0] for file in files})

    def get_fucci_red_image_path(self, file):
        return os.path.join(self.data_set_dir, f"{file}-ch1.tif")

    def get_fucci_green_image_path(self, file):
        return os.path.join(self.data_set_dir, f"{file}-ch2.tif")

    def get_dapi_image_path(self, file):
        return os.path.join(self.data_set_dir, f"{file}-ch3.tif")
