import darknet
import cv2

def load(img):
    """automatically load image from str or array"""
    if isinstance(img, str):
        src = cv2.imread(img)
    elif isinstance(img, np.ndarray):
        src = img
    else:
        raise Exception('helpers.load Error: Wrong input.')
    if src is None:
        raise Exception('autoaim.aimmat Error: Image loading failed.')
    return src

def crop(img, box):
    (x, y, w, h) = box
    (x1, y1, x2, y2) = (max(0,x),max(0,y),min(x+w,img.shape[1]),min(y+h,img.shape[0]))
    return img[y1:y2,x1:x2]

def peek(img, timeout=0, update=False):
    """an easy way to show image, return img"""
    cv2.imshow('showoff', img)
    key = cv2.waitKey(timeout)
    return img


def showoff(img, timeout=0, update=False):
    """an easy way to show image, return `exit`"""
    cv2.imshow('showoff', img)
    key = cv2.waitKey(timeout)
    if not update:
        cv2.destroyAllWindows()
    if key == 27:
        return True
    return False

def time_this(original_function):
    """timing decorators for a function"""
    print("decorating")

    def new_function(*args, **kwargs):
        print("starting timer")
        import datetime
        before = datetime.datetime.now()
        x = original_function(*args, **kwargs)
        after = datetime.datetime.now()
        print("Elapsed Time = {0}".format(after-before))
        return x
    return new_function

def video_load(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError("Couldn't open video!")
    info = {
        'rate': int(cap.get(5)),
        'frames': int(cap.get(7))
        }
    return (cap, info)