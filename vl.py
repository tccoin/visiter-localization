import cv2
import numpy as np
import darknet
import time
import helper
import pymynt

darknet.set_gpu(0)

MODEL_MEAN_VALUES= (78.4263377603, 87.7689143744, 114.895847746)
age_list=[2,6,10,15,20,40,50,70]
label_id = 0
# age_list=['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
gender_list=['Male','Female']
font = cv2.FONT_HERSHEY_SIMPLEX
# font = ImageFont.truetype(font='font/FiraMono-Medium.otf', size=20)

def agcnn_load(method='opencv'):
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
    result = {'method':'opencv'}
    blob = cv2.dnn.blobFromImage(face_img, 1,(227, 227), MODEL_MEAN_VALUES, swapRB=False)

    #PredictGender
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = gender_list[gender_preds[0].argmax()]
    result['gender'] = gender
    # result['gender_score'] = gender_preds[0].argmax()

    #PredictAge
    age_net.setInput(blob)
    age_preds = age_net.forward()
    # result['age'] = age_list[age_preds[0].argmax()]
    # result['age_score'] = age_preds[0].argmax()
    result['age'] = (np.array(age_list)*age_preds[0]).sum()
    return result


def agcnn_detect(image, yolo_result, models, origin=(0,0), margin=0, method='opencv'):
    faces = agcnn_face(image, models[-1])
    if len(faces)>0:
        m = margin
        (x, y, w, h) = faces[0]
        box = (x-m, y-m, w+2*m, h+2*m)
        image = helper.crop(image, box)
        result = agcnn_predict(image, *(models[0:-1]), method=method)
        result['box'] = (x+origin[0], y+origin[1], w, h)
        result['confidence'] = w*h
        if 'face' in yolo_result:
            lr = yolo_result['face'] #last_result
            cr = result #current_result
            _1 = cr['confidence']+lr['confidence']
            # age
            _2 = cr['age']*cr['confidence']+lr['age_average']*lr['confidence']
            result['age_average'] = _2/_1
            # gender
            _3 = 1 if cr['gender']==gender_list[0] else -1
            _4 = 1 if lr['gender_average']==gender_list[0] else -1
            _5 = _3*cr['confidence']+_4*lr['confidence']
            result['gender_average'] = gender_list[_5<0]
            # confidence
            result['confidence'] = _1
        else:
            result['age_average'] = result['age']
            result['gender_average'] = result['gender']
        return result
    else:
        if 'face' in yolo_result:
            return yolo_result['face']
        else:
            return None

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
        overlay_text= "{} {:.1f}".format(r['gender_average'], r['age_average'])
        cv2.rectangle(image, (x, y), (x+w, y+h),(255, 0, 0), 2)
        cv2.putText(image, overlay_text, (x, y),font, 1.5, (0,100,200), 3, cv2.LINE_AA)

def yolo_draw(image, results):
    for r in results:
        (x,y,w,h) = r['box']
        # overlay_text= "{} {:.2f} {}".format(r['class'], r['score'], r['id'])
        overlay_text= "{}".format(r['id'])
        if r.get('deleted', False):
            cv2.rectangle(image, (x, y), (x+w, y+h),(255, 0, 0), 2)
        else:
            cv2.rectangle(image, (x, y), (x+w, y+h),(255, 255, 0), 2)
        cv2.putText(image, overlay_text, (x, y),font, 1.2, (0,0,0), 2, cv2.LINE_AA)

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

        # face = agcnn_predict(face_img, *agcnn_models[0:-1], method=method)
        face = agcnn_detect(face_img, r, agcnn_models,origin=box[0:2], method=method, margin=15,)
        if face is None:
            continue
        else:
            r['face'] = face
            results += [face]
    return results

def yolo_postprocessing(results, safe_region=None):
    results = [r for r in results if r['class']=='person']
    if safe_region is not None:
        boxes = [r['box'] for r in results]
        for box in boxes:
            box[1] = max(0, box[1])
            box[3] = min(safe_region[3]-box[1], box[3])
        def _include(a,b):
            (x1,y1,w1,h1) = a
            (x2,y2,w2,h2) = b
            return x1>=x2 and x1+w1<=x2+w2# and y1>=y2 and y1+h1<=y2+h2
        results = [r for r in results if _include(r['box'], safe_region)]
    results.sort(key=lambda r: r['box'][1], reverse=True)
    results.sort(key=lambda r: r['box'][0], reverse=True)
    i = -1
    while True:
        i += 1
        if i >= len(results)-1:
            break
        ri = results[i]
        (x1,y1,w1,h1) = ri['box']
        for j in range(i+1, len(results)):
            rj = results[j]
            (x2,y2,w2,h2) = rj['box']
            if x2+w2<=x1:
                continue
            if x2+w2>=x1+w1 and y2+h2>=y1+h1:
                results.remove(ri)
                i -= 1
                break
    return results

def label(last, current, default_life=25):
    def generate_id(box):
        global label_id
        (x,y,w,h) = box
        if w*h>15*15:
            label_id += 1
            return label_id
        else:
            return 0

    last = [x for x in last if x['id_life']>0]
    if len(last)==0:
        for c in current:
            c['id'] = generate_id(c['box'])
    else:
        _last = last.copy()
        _current = current.copy()

        while True:
            # data for calculating scores
            cc = np.array([x['box'] for x in _current]).copy()
            ll = np.array([x['box'] for x in _last]).copy()
            if cc.size>0:
                cc[:,0:2] = cc[:,0:2]+cc[:,2:4]/2
            if ll.size>0:
                ll[:,0:2] = ll[:,0:2]+ll[:,2:4]/2
            # calculate scores
            for i in range(len(_current)):
                if len(ll)==0:
                    _current[i]['id'] = generate_id(cc[i])
                    _current[i]['match_score'] = 0
                    continue
                _1 = (ll-cc[i])
                _2 = (_1*_1)[:,0:2].sum(1)
                scores = _2/ll[:,2]/ll[:,3]*10000
                match_index = scores.argmin()
                _current[i]['match_index'] = match_index
                _current[i]['match_score'] = scores[match_index]
            # sort
            _current.sort(key=lambda x:x['match_score'])
            # match
            matched_c = []
            matched_l = []
            for i in range(len(_current)):
                match_index = _current[i]['match_index']
                if not match_index in matched_l:
                    _current[i]['id'] = _last[match_index]['id']
                    if 'face' in _last[match_index]:
                        _current[i]['face'] = _last[match_index]['face']
                    matched_c.append(i)
                    matched_l.append(match_index)
            # delete matched items
            matched_c.sort(reverse=True)
            matched_l.sort(reverse=True)
            for i in matched_c:
                _current.pop(i)
            for i in matched_l:
                _last.pop(i)
            # loop condition
            if len(_last)==0 and len(_current) !=0:
                for c in _current:
                    c['id'] = generate_id(c['box'])
                break
            if len(_current)==0:
                break
        last = _last
    for x in last:
        x['id_life'] -= 1
    for x in current:
        x['id_life'] = default_life
    return last,current

if __name__ == "__main__":

    ### LOAD ###
    agcnn_models_opencv = agcnn_load(method='opencv')
    yolo_models = yolo_load()
    count = 0
    last_yolo_results = []
    yolo_results = []
    agcnn_results = []
    depth = None

    ### VIDEO ###
    cap, info = helper.video_load('data/left.mp4')
    cap.set(1,600)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

    ### IMAGE ###
    # while True:
    #     image = cv2.imread('data/web.jpeg')

    ### IMAGE-RGBD ###
    # for i in range(1,192):
    #     image = cv2.imread('data/mynt/left_{}.jpg'.format(i))

    ### CAMERA ###
    # pymynt.init_camera('raw')
    # while True:
        # depth = pymynt.get_depth_image()
        # if depth.shape[0] < 10:
        #     continue
        # depth_mat = depth/np.max(depth)*255
        # depth_mat = depth_mat.astype('uint8')
        # depth_mat = cv2.cvtColor(depth_mat, cv2.COLOR_GRAY2RGB)
        # image = pymynt.get_left_image()
        # helper.peek(image,1)

        ### DETECT ###
        if True:
            tic = time.time()
            # yolo
            yolo_results = yolo_detect(image, *yolo_models)
            yolo_results = yolo_postprocessing(yolo_results, (10,0,image.shape[1]-20,image.shape[0]))
            last_yolo_results, yolo_results = label(last_yolo_results, yolo_results)
            last_yolo_results += yolo_results
            # agcnn
            agcnn_results = agcnn_detect_from_yolo(image, yolo_results, agcnn_models_opencv,method='opencv')
            toc = time.time()
            result_image = image.copy()
        else:
            result_image = image

        ### OUTPUT ###
        print('frame: {:0>4d} fps: {:.1f}'.format(count, 1/(toc-tic)))
        yolo_draw(result_image,yolo_results)
        agcnn_draw(result_image,agcnn_results)
        helper.peek(result_image,1)
        cv2.imwrite('data/result/result{}.jpg'.format(count), result_image)
        count = count+1