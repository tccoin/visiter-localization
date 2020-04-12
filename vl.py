import cv2
import numpy as np
import darknet
import time
import helper
import caffe
import pymynt

caffe.set_mode_gpu()
caffe.set_device(0)
darknet.set_gpu(0)

MODEL_MEAN_VALUES= (78.4263377603, 87.7689143744, 114.895847746)
age_list=['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
gender_list=['Male','Female']
font = cv2.FONT_HERSHEY_SIMPLEX
# font = ImageFont.truetype(font='font/FiraMono-Medium.otf', size=20)

def agcnn_load(method='opencv'):
    if method == 'caffe':
        age_net = caffe.Classifier('agcnn/deploy_age.prototxt', 'agcnn/age_net.caffemodel',
                            mean=np.array(MODEL_MEAN_VALUES),
                            channel_swap=(1,0,2),
                            raw_scale=255,
                            image_dims=(256, 256)
                            )
        gender_net = caffe.Classifier('agcnn/deploy_gender.prototxt', 'agcnn/gender_net.caffemodel',
                            mean=np.array(MODEL_MEAN_VALUES),
                            channel_swap=(1,0,2),#bgr rbg
                            raw_scale=255,
                            image_dims=(256, 256))
        face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_alt.xml')
        return(age_net,gender_net,face_cascade)
    elif method == 'opencv':
        age_net = cv2.dnn.readNetFromCaffe('agcnn/deploy_age.prototxt', 'agcnn/age_net.caffemodel')
        gender_net= cv2.dnn.readNetFromCaffe('agcnn/deploy_gender.prototxt', 'agcnn/gender_net.caffemodel')
        age_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        age_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        gender_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        gender_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_alt.xml')
        return(age_net,gender_net,face_cascade)

def agcnn_face(image, face_cascade):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    return face_cascade.detectMultiScale(gray,1.1, 5)

def agcnn_predict(face_img, age_net, gender_net, method='opencv'):
    if method == 'caffe':
        result = {'method':'caffe'}
        _w = image.shape[0]
        _h = image.shape[1]
        if _w>_h:
            h = 227
            w = int(h/_h*_w)
            (x, y) = (int((_w-227)/2), 0)
            face_img = cv2.resize(face_img,(w,h))[y:y+h,x:x+w]
        elif _w>_h:
            w = 227
            h = int(w/_w*_h)
            (x, y) = (0, int((_h-227)/2))
            face_img = cv2.resize(face_img,(w,h))[y:y+h,x:x+w]
        else:
            face_img = cv2.resize(face_img,(227,227))
        # helper.peek(face_img)

        # face_img = cv2.cvtColor(face_img,cv2.COLOR_BGR2RGB)
        face_img = face_img.swapaxes(0, 2).swapaxes(1, 2)

        #PredictGender
        gender_net.blobs['data'].data[...] = [face_img]
        prediction = gender_net.forward()['prob']
        result['gender'] = gender_list[prediction[0].argmax()]
        result['gender_score'] = prediction[0].max()

        #PredictAge
        age_net.blobs['data'].data[...] = [face_img]
        prediction = age_net.forward()['prob']
        result['age'] = age_list[prediction[0].argmax()]
        result['age_score'] = prediction[0].max()

        return result
    elif method == 'opencv':
        result = {'method':'opencv'}
        blob = cv2.dnn.blobFromImage(face_img, 1,(227, 227), MODEL_MEAN_VALUES, swapRB=False)

        #PredictGender
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = gender_list[gender_preds[0].argmax()]
        result['gender'] = gender
        result['gender_score'] = gender_preds[0].argmax()

        #PredictAge
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = age_list[age_preds[0].argmax()]
        result['age'] = age
        result['age_score'] = age_preds[0].argmax()
        return result


def agcnn_detect(image, models, margin=0, method='opencv'):
    faces = agcnn_face(image, models[-1])
    if len(faces)>0:
        m=margin
        (x, y, w, h) =faces[0]
        box = (x-m, y-m, w+2*m, h+2*m)
        image = helper.crop(image, box)
    result = agcnn_predict(image, *(models[0:-1]), method=method)
    if len(faces)>0:
        result['box'] = (x, y, w, h)
    else:
        result['box'] = (0, 0, image.shape[0],image.shape[1])
    return [result]

def yolo_load():
    net = darknet.load_net(b"yolo/cfg/yolov3.cfg", b"yolo/weights/yolov3.weights", 0)
    meta = darknet.load_meta(b"yolo/cfg/coco.data")
    # net = darknet.load_net(b"yolo/cfg/yolov3-tiny.cfg", b"yolo/weights/yolov3-tiny.weights", 0)
    # meta = darknet.load_meta(b"yolo/cfg/coco.data")
    return (net,meta)

def yolo_detect(image, net, meta):
    results = []

    # EncodeToDarknetImage
    # im = helper.array_to_image(image)
    # darknet.rgbgr_image(image)

    # YoloDetect
    yolo_result = darknet.detect(net, meta, image)
    for objection in yolo_result:
        class_name, class_score, (x, y, w, h) = objection
        [x, y, w, h] = [x-w/2,y-h/2,w,h]
        [x, y, w, h] = [int(x) for x in [x, y, w, h]]
        results.append({
            'box':[x, y, w, h],
            'class':class_name.decode(),
            'score':class_score
        })
    return results

def agcnn_draw(image, results):
    for r in results:
        (x,y,w,h) = r['box']
        overlay_text= "{} {}".format(r['gender'], r['age'])
        if r['method']=='caffe':
            cv2.putText(image, overlay_text, (x, y+30),font, 0.7, (50,50,50), 2, cv2.LINE_AA)
            cv2.rectangle(image, (x, y), (x+w, y+h),(255, 255, 0), 2)
        else:
            cv2.putText(image, overlay_text, (x, y+50),font, 0.7, (100,100,0), 2, cv2.LINE_AA)
            cv2.rectangle(image, (x, y), (x+w, y+h),(255, 255, 0), 2)

def yolo_draw(image, results):
    for r in results:
        (x,y,w,h) = r['box']
        overlay_text= "{} {:.2f}".format(r['class'], r['score'])
        cv2.putText(image, overlay_text, (x, y+h),font, 0.7, (0,0,0), 2, cv2.LINE_AA)
        cv2.rectangle(image, (x, y), (x+w, y+h),(255, 255, 0), 2)

def agcnn_detect_from_yolo(image, yolo_results, agcnn_models,method='opencv'):
    results = []
    count = 0
    for r in yolo_results:
        (x, y, w, h) = r['box']
        m = 20 #margin
        (x, y, w, h) = (x+w/4, y, w/2, h)
        (x, y, w, h) = (x-m, y-m, w+2*m, w+2*m)
        box = [int(x) for x in (x, y, w, h)]
        face_img = helper.crop(image, box)
        count = (count+1)%5
        # cv2.imshow('face'+str(count),face_img)

        if face_img.size==0:
            continue
        
        face = agcnn_predict(face_img, *agcnn_models[0:-1], method=method)
        # face = agcnn_detect(face_img, agcnn_models, method=method, margin=15)[0]
        if 'box' in face:
            (_x, _y, _w, _h) = face['box']
            face['box'] = [int(x) for x in (_x+x, _y+y, _w, _h)]
        else:
            face['box'] = [int(x) for x in (x, y, w, h)]
        results += [face]
    return results


if __name__ == "__main__":
    cap, info = helper.video_load('data/left.mp4')
    agcnn_models_opencv = agcnn_load(method='opencv')
    # agcnn_models_caffe = agcnn_load(method='caffe')
    yolo_models = yolo_load()
    pymynt.init_camera('raw')
    # for i in range(info['frames']):
    cap.set(1,600)
    count = 0
    yolo_results = []
    agcnn_results = []
    # while True:
    while cap.isOpened():
    # for i in range(1,10):

        ### DATA ###
        success, image = cap.read()
        if not success:
            break
        # image = cv2.imread('data/img01-1.jpg')

        ### MYNT ###
        # depth = pymynt.get_depth_image()
        # if depth.shape[0] < 10:
        #     continue
        # depth_mat = depth/np.max(depth)*255
        # depth_mat = depth_mat.astype('uint8')
        # depth_mat = cv2.cvtColor(depth_mat, cv2.COLOR_GRAY2RGB)
        # image = pymynt.get_left_image()
        # helper.peek(image,1)

        ### DETECT ###
        count = (count+1)%1
        if count == 0:
            tic = time.time()
            yolo_results = yolo_detect(image, *yolo_models)
            # yolo_results = [*filter(lambda r:r['class']=='person', yolo_results)]
            agcnn_results = []
            # agcnn_results += agcnn_detect(image, agcnn_models_opencv,method='opencv')
            # agcnn_results += agcnn_detect(image, agcnn_models_caffe,method='caffe')
            agcnn_results += agcnn_detect_from_yolo(image, yolo_results, agcnn_models_opencv,method='opencv')
            # agcnn_results += agcnn_detect_from_yolo(image, yolo_results, agcnn_models_caffe,method='caffe')
            # agcnn_results = agcnn_detect(image, *agcnn_models)
            toc = time.time()
            print('fps: {:.1f}'.format(1/(toc-tic)))
            result_image = image.copy()
        else:
            result_image = image
        yolo_draw(result_image,yolo_results)
        agcnn_draw(result_image,agcnn_results)
        helper.peek(result_image,1)