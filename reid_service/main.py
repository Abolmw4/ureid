from reidentification_service import Ureid, WriteVideo
from reidentification_util import load_json_file
from multiprocessing import Queue


def main():
    config = load_json_file("../configs/config_service.json")
    queue = Queue()

    ureidProc = Ureid(reid_model_src=config["reid_model"], yolo_model_src=config["yolo_model"], device='cpu', vidoe_or_image_src=config["vidoe_src"], queue=queue)
    wv = WriteVideo(queue=queue)

    ureidProc.start()
    wv.start()

    ureidProc.join()
    wv.join()

if __name__ == "__main__":
    main()
