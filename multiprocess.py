from multiprocessing import Process, Queue, Manager
from shared_arrays import ArrayQueue
import vl
import time
import helper

def read_image(image_queue,yolo_results,agcnn_results,timeout=0.01,skip=10):
    cap, info = helper.video_load('data/left.mp4')
    cap.set(1, 500)
    count = 0
    init = 0
    while cap.isOpened():
        tic = time.time()
        success, image = cap.read()
        if not success:
            break
        if count%(skip+1) == 0:
            image_queue.clear()
            image_queue.put(image)
        if len(yolo_results)==0:
            if not init:
                time.sleep(3)
        else:
            init = True
            vl.yolo_draw(image,yolo_results)

        if not len(agcnn_results)==0:
            vl.agcnn_draw(image,agcnn_results)

        toc = time.time()
        sleep = (timeout-(toc-tic))*1000
        if sleep < 0:
            sleep = 0
        count+=1
        helper.peek(image, int(sleep+1))

def yolo(yolo_image_queue, yolo_results_queue, agcnn_image_queue,yolo_results):
    yolo_models = vl.yolo_load()
    count = 0
    while True:
        image = yolo_image_queue.get()
        results = vl.yolo_detect(image, *yolo_models)
        yolo_results[:] = []
        yolo_results += results


        if yolo_results_queue.empty():
            yolo_results_queue.put(results,False)
        if agcnn_image_queue.empty():
            agcnn_image_queue.put(image)

        count+=1
        print('yolo ',count,len(yolo_results))

def agcnn(agcnn_image_queue, yolo_results_queue, agcnn_results):
    agcnn_models = vl.agcnn_load()
    count = 0
    while True:
        image = agcnn_image_queue.get()
        yolo_results = yolo_results_queue.get()
        results = vl.agcnn_detect_from_yolo(image, yolo_results, *agcnn_models)
        agcnn_results[:] = []
        agcnn_results += results
        count+=1
        print('agcnn ',count)

if __name__ == '__main__':

    yolo_image_queue = ArrayQueue(50, 5)
    yolo_results_queue = Queue(1)
    agcnn_image_queue = ArrayQueue(50, 1)

    manager = Manager()
    yolo_results = manager.list()
    agcnn_results = manager.list()

    processes = []
    processes += [Process(
        target=read_image,
        args=[
            yolo_image_queue,yolo_results,agcnn_results
        ])]
    processes += [Process(
        target=yolo,
        args=[
            yolo_image_queue, yolo_results_queue, agcnn_image_queue,yolo_results
        ])]
    processes += [Process(
        target=agcnn,
        args=[
            agcnn_image_queue, yolo_results_queue, agcnn_results
        ])]
    for process in processes:
        process.start()
    for process in processes:
        process.join()