import os.path
from html.parser import piclose
from multiprocessing import Process, Queue
from datetime import datetime
from typing import Union, List, Any
import numpy as np
import torch.cuda
import torchvision.transforms as T
from reid_service.reidentification_util import load_json_file
from reidentification_util import LOG
from ultralytics import YOLO
import cv2
import mediapipe as mp
import pickle

CONFIG_SRC = "../configs/config_service.json"

class Ureid(Process):
    test_transformer = T.Compose([T.ToTensor(), T.Resize((256, 128), interpolation=3), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
    config = load_json_file(CONFIG_SRC)
    def __init__(self, yolo_model_src: str, reid_model_src: str, vidoe_or_image_src, device, queue: Queue):
        super().__init__()
        self.yolo_model_src = yolo_model_src
        self.reid_model_src = reid_model_src
        self.device = device
        self.video_or_image_src = vidoe_or_image_src
        self.queue = queue
        self.gallery: List[List[Any]] = []
        self.key_points = [11, 12, 23, 24, 27, 28] # most important landmarks
        self.mp_pose = mp.solutions.pose # for pose estimation (good or bad person)

    def __del__(self):
        print("Ureid object deleted")

    @property
    def device(self) -> str:
        return self.__device
    @device.setter
    def device(self, value: str) -> None:
        self.__device = 'cuda' if value in ['cpu', 'cuda'] and value == 'cuda' and torch.cuda.is_available() else 'cpu'

    @property
    def reid_model_src(self) -> str:
        return self.__reid_model_src

    @reid_model_src.setter
    def reid_model_src(self, value: str) -> None:
        if isinstance(value, str) and os.path.isfile(os.path.join(os.getcwd(), value)) and value.split('.')[-1] == 'pt':
            self.__reid_model_src = os.path.join(os.getcwd(), value)
        else:
            raise ValueError(f"model adress{value} not valid!")

    @property
    def yolo_model_src(self):
        return self.__yolo_model_src

    @yolo_model_src.setter
    def yolo_model_src(self, value: str):
        if isinstance(value, str) and os.path.isfile(os.path.join(os.getcwd(), value)) and value.split('.')[-1] == 'pt':
            self.__yolo_model_src = os.path.join(os.getcwd(), value)
        else:
            raise ValueError(f"model adress{value} not valid!")

    @property
    def video_or_image_src(self) -> Union[str, int]:
        return self.__video_or_image_src

    @video_or_image_src.setter
    def video_or_image_src(self, value: str) -> None:
        cls = type(self)
        if (
                value is not None
                and os.path.isfile(value)
                and value.split(".")[-1] in cls.config["video_format"]
        ):
            self.__video_or_image_src = value

        elif value is not None and os.path.isdir(value):
            self.__video_or_image_src = value

        elif (
                value is not None
                and os.path.isfile(value)
                and value.split(".")[-1] not in cls.config["video_format"]
        ):
            raise ValueError(f"'video_src is not valid!'")
        else:
            self.__video_or_image_src = 0

    def __load_model(self):
        try:
            reid_model = torch.jit.load(self.reid_model_src).to(self.device)
            reid_model.eval()
            yolo_model = YOLO(self.yolo_model_src).to(self.device)
            LOG.debug("yolo and reid model load successfully")
            return yolo_model, reid_model
        except Exception as error:
            LOG.error(f"{error}")

    def __check_good_or_bad_person(self, person: np.ndarray, score: float) -> bool:

        with self.mp_pose.Pose(static_image_mode=True) as pose:
            results = pose.process(person)
            scr = 0
            try:
                for i in self.key_points:
                    if results.pose_landmarks.landmark[i].visibility > score:
                        scr += 1
                if scr == len(self.key_points):
                    LOG.debug("Good person")
                    return True
                else:
                    LOG.debug("Bad person")
                    return False
            except AttributeError:
                LOG.info("Bad person")
                return False

    def __extract_feature(self, person: np.ndarray, reid_model) -> torch.Tensor:
        cls = type(self)
        try:
            return reid_model(cls.test_transformer(person).reshape((1, 3, 256, 128)).to(self.device))
        except Exception as error:
            LOG.error(f"{error}")


    def __calculate_distance(self, query: torch.Tensor, gallery: torch.Tensor) -> float:
        cosi = torch.nn.CosineSimilarity()
        cosi_value = cosi(query, gallery).item() * 100
        return cosi_value


    def assign_id(self, person: List[Any], threshold: float) -> List[Any]:
        cls = type(self)
        max_smilarity: float = 0.0
        id = None
        qfeature, qimage, qid, qtime = person
        for indx, gitem in enumerate(self.gallery):
            if self.__calculate_distance(query=qfeature, gallery=gitem[0]) > threshold:
                LOG.debug("Query_id:{qid} == Gallery_id:{gid}".format(qid=qid, gid=gitem[-2]))
                max_smilarity = self.__calculate_distance(query=qfeature, gallery=gitem[0])
                id = gitem[-2]
                LOG.debug("Gallery_id:{gid} - date:{gdate} ---updated to--> date:{qdate}".format(gid=gitem[-2], gdate=gitem[-1], qdate=qtime))
                self.gallery[indx][-1] = qtime

        if max_smilarity == 0.0 and id is None:
            LOG.debug("Query_id:{qid} not in Gallery list | gallery size:{gsize}".format(qid=qid, gsize=len(self.gallery)))
            self.gallery.append(person)
            LOG.debug("Query_id:{qid} inserted to gallery list | gallery size:{gsize}".format(qid=qid, gsize=len(self.gallery)))
            id = qid

        now = list(map(lambda input: int(input), datetime.now().strftime("%y %m %d %H %M %S").split(' ')))
        for indx, gitem in enumerate(self.gallery):
            out = [now[0] - gitem[-1][0], now[1] - gitem[-1][1], now[2] - gitem[-1][2], now[3] - gitem[-1][3], now[4] - gitem[-1][4], now[5] - gitem[-1][5]]
            if out[0] != 0 or out[1] != 0 or out[2] != 0:
                LOG.debug("Gallery_id:{gid} delete from gallery list".format(gid=gitem[-2]))
                del self.gallery[indx]
            elif out[-3] * 60 + out[-2] > cls.config["item_elapsed_gallery"]:
                LOG.debug("Gallery_id:{gid} delete from gallery list".format(gid=gitem[-2]))
                del self.gallery[indx]
        return [qfeature, qimage, id, qtime]

    def run(self):
        LOG.info("******* Ureid up *********")
        cls = type(self)
        yolo_net, reid_net = self.__load_model()  # load model
        cap = cv2.VideoCapture(self.video_or_image_src)
        if not cap.isOpened():
            LOG.error("Error Opening video")
        id, fram = 0, 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # detect person from each frame
                result = yolo_net.predict(frame, classes=[0], show=False)
                # crop person from each frame and extract feature
                LOG.info(f"{len(result[0].boxes.xyxy.cpu().tolist())} person detected")
                for idx, item in enumerate(result[0].boxes.xyxy.cpu().tolist()):
                    crop_image = frame[int(item[1]) : int(item[3]), int(item[0]) : int(item[2])]
                    # check this person good or not and then inset to queue
                    if self.__check_good_or_bad_person(person=crop_image, score=cls.config["visibility_score_for_body_landmarks"]):
                        id += 1
                        if len(self.gallery) == 0:
                            LOG.debug("Gallery list is empty")
                            time = list(map(lambda input: int(input), datetime.now().strftime("%y %m %d %H %M %S").split(' ')))
                            self.gallery.append([self.__extract_feature(person=crop_image, reid_model=reid_net), crop_image, id, time])
                            self.queue.put(pickle.dumps({"id": id, "frame": frame, "cordinate":item}))
                        else:
                            LOG.debug(f"Gallery List not empty | gallery size:{len(self.gallery)}")
                            time = list(map(lambda input: int(input), datetime.now().strftime("%y %m %d %H %M %S").split(' ')))
                            result_of_assignment = self.assign_id([self.__extract_feature(person=crop_image, reid_model=reid_net), crop_image, id, time], threshold=cls.config["similarity_threshold"])
                            self.queue.put(pickle.dumps({"id":result_of_assignment[-2], "frame":frame, "cordinate": item}))
                            LOG.debug(f"id:{result_of_assignment[-2]} add to queue")
                    else:
                        self.queue.put(pickle.dumps({"id": "Unknown", "frame": frame, "cordinate": item}))
                        LOG.debug("id Unknown add to queue")
        LOG.warning("******* Ureid Down *********")
        

class WriteVideo(Process):
    def __init__(self):
        super().__init__()
        pass

