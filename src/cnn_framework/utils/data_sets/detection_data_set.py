import json
from abc import abstractmethod

from .abstract_data_set import AbstractDataSet


class DetectionDataSet(AbstractDataSet):
    @abstractmethod
    def generate_raw_images(self, filename):
        raise NotImplementedError

    @staticmethod
    def get_image_id(annotations, filename):  # extension not included
        for image in annotations["images"]:
            if image["file_name"] == filename:
                return image["id"]
        raise ValueError(f"Image {filename} not found in JSON annotation file.")

    def get_target(self, annotations, image_id):
        boxes, labels = [], []
        for annotation in annotations["annotations"]:
            if annotation["image_id"] != image_id:
                continue
            boxes.append(annotation["bbox"])
            labels.append(annotation["category_id"])
        return {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
        }

    def get_image_file_name_from_id(self, image_id):
        """
        Returns file name without extension.
        """
        bounding_boxes_reader = self.data_manager.get_bounding_boxes_path()
        json_reader = open(bounding_boxes_reader)
        annotations = json.load(json_reader)

        # Get image path from image_id
        for image in annotations["images"]:
            if image["id"] == image_id:
                return image["file_name"]
        raise ValueError(f"Image {image_id} not found in JSON annotation file.")

    def __getitem__(self, idx):
        _, raw_img_input, raw_img_output, _ = super().__getitem__(idx)

        # Image index and additional data are ignored for train_one_epoch
        return raw_img_input, raw_img_output
