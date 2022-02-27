# Activate tf-env before using this script
# link to install the env 
# Follow the directory path, change the dir if necessary


import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import glob
from tensorflow.keras.preprocessing import image
import time
from object_detection.utils import label_map_util
import cv2
import webcolors
import warnings
warnings.filterwarnings('ignore')

def load_img_class_model(dir=r'models\classification_model\Tomogram_CNN_model_epoch=50.h5'):
    new_model = tf.keras.models.load_model(r'{}'.format(dir))
    return new_model

def load_obj_model():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    PATH_TO_MODEL_DIR = 'models/object_detection_model/'
    PATH_TO_LABELS = 'models/object_detection_model/label_map.pbtxt'
    MIN_CONF_THRESH = float(0.5)
    PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR
    print('Loading model...', end='')
    start_time = time.time()
    detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Done! Took {} seconds'.format(elapsed_time))
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)
    #new_model = tf.keras.models.load_model(r'{}'.format(dir))
    return detect_fn


def load_images(dir):
    a = glob.glob('{}'.format(dir))[1]
    if 'img' in a.split('.png')[0]:
        files = sorted(glob.glob('{}'.format(dir)),key=lambda name: int(name.split("_img")[-1].split(".")[0]))
    if 'Hist' in a.split('.png')[0]:
        files = sorted(glob.glob('{}'.format(dir)),key=lambda name: int(name.split("Hist")[-1].split(".")[0]))
    return files

def get_Focus(filename,classi_model):
    img1 = image.load_img(filename,target_size=(150,150))
    z = int(filename.split("img")[-1].split(".")[0])
    f = filename.split("img")
    Y = image.img_to_array(img1)/255
    X = np.expand_dims(Y,axis=0)  
    val = classi_model.predict_classes(X)
    if val == 0:
        return "infocus", z
    elif val == 1:
        return "outfocus", z 
    '''
    else:
        return "confused", z
    '''

def get_zplane(files,classi_model,infocus=False):
    z_plane = []
    focus_label = []

    if infocus==False:
        focus_label = []
        z_plane = []
        for pic in files:
            f, z= get_Focus(pic,classi_model)
            focus_label.append(f)
            z_plane.append(z)
        return  focus_label,z_plane
    if infocus==True:
        file_infocus = []
        z_in = []
        for pic in files:
            f, z= get_Focus(pic,classi_model)
            z_plane.append(z)
            focus_label.append(f)
            for i,k in enumerate(z_plane):
                if focus_label[i] == 'infocus':
                    if (int(pic.split('img')[1].split('.')[0]) == k):
                        #print('Done:{}'.format(pic))
                        file_infocus.append(pic)
                        z_in.append(int(pic.split('img')[1].split('.')[0]))
        return z_in,file_infocus

def get_best_image(files,classi_model):
    def load_image(filename):
        img1 = image.load_img(filename,target_size=(150,150))
        z = int(filename.split("img")[-1].split(".")[0])
        Y = image.img_to_array(img1)/255
        X = np.expand_dims(Y,axis=0)
        val.append(classi_model.predict(X))
        z_p.append(z)
        vaL_z_dict = dict(zip(z_p,val)) 
        return vaL_z_dict
    val = []
    z_p = []
    vaL_z_dict = {}
    f = 0
    files = files
    for pic in files:
        val_z_dict = load_image(pic)
        
    best_pic = min(val_z_dict,key=val_z_dict.get)
    for pic in files:
        a = pic.split('img')[1].split('.')[0]
        #a = pic.split('Hist')[1].split('.')[0]
        
        if a == str(best_pic):
            f = pic
    print('best z_plane='+str(best_pic))
    return [f]


def detect_obj(files,classi_model):
    z =[]
    count=[]
    files = get_best_image(files,classi_model)
    MIN_CONF_THRESH = float(0.5)
    for file in files:
        print('Running inference for {}... '.format(file), end='')
        z.append(int(file.split("_img")[-1].split(".")[0]))
        image = cv2.imread(file)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        imH, imW, _ = image.shape
        image_expanded = np.expand_dims(image_rgb, axis=0)

        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        input_tensor = tf.convert_to_tensor(image)
        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis, ...]

        # input_tensor = np.expand_dims(image_np, 0)
        detect_fn = load_obj_model()
        detections = detect_fn(input_tensor)

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
        scores = detections['detection_scores']
        boxes = detections['detection_boxes']
        classes = detections['detection_classes']

        d = 0
        
        for i in range(len(scores)):
            if ((scores[i] > MIN_CONF_THRESH) and (scores[i] <= 1.0)):
                d += 1
                count.append(d)
                num_of_obj = max(count)
        print('Detected_number_of_objects:{}'.format(num_of_obj))
    return num_of_obj,z,files


def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

def get_colour_name(requested_colour):
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
        actual_name = None
    return closest_name


def image_segmentation(files,plot=False,dir=r'models\classification_model\Tomogram_CNN_model_epoch=50.h5'):
    #dir = r'C:\Users\clive\PycharmProjects\Research\Neural Networks\Tomogram_CNN_model.h5'
    model = load_img_class_model(dir)
    files = get_best_image(files,model);
    for image in files:
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pixel_vals = img.reshape((-1,3))
        pixel_vals = np.float32(pixel_vals)
        
        c = True
        k = 0
        while c:
            closest_name = []
            k= k+1
            #print('k='+str(k))
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
            retvcal, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            centers = np.uint8(centers)
            segmented_data = centers[labels.flatten()]
            segmented_image = segmented_data.reshape((img.shape))
            label = []
            a = labels.tolist()
            percent=[]
            centroid = centers
            for i in range(len(a)):
                label.append(a[i][0])

            for i in range(len(centroid)):
                j=label.count(i)
                j=j/(len(label))
                percent.append(j)

            for i in range(len(centroid)): 
                requested_colour = (centroid[i][0],centroid[i][1],centroid[i][2])
                closest_name.append(get_colour_name(requested_colour))
            for name in closest_name:
                #print(closest_name)
                if 'blue' in name:      
                    c = False
                    #print(name)
                else:
                    #print(name)
                    #k+=1
                    continue 
                    
    for i in centers:
        a = get_colour_name(i)
        if 'blue' in a:
            b_c = i
    
    l,w,_ = segmented_image.shape
    for i in range(l):
        for k in range(w):
            if (segmented_image[i][k][0],segmented_image[i][k][1],segmented_image[i][k][2]) != (b_c[0],b_c[1],b_c[2]):
                (segmented_image[i][k][0],segmented_image[i][k][1],segmented_image[i][k][2]) = (0,0,0)
    kernel = np.ones((50,50),np.uint8)
    segmented_image =  cv2.morphologyEx(segmented_image, cv2.MORPH_OPEN, kernel)
    tight = cv2.Canny(segmented_image, 240, 250)
    contours,h = cv2.findContours(tight, 
                            cv2.RETR_LIST,
                            cv2.CHAIN_APPROX_NONE)
    test1 = img.copy()
    cv2.drawContours(test1,contours,-1,(0,255,0), 3);
    print('Finished Segmenting image')
    if plot==True:
        plt.subplot(131),plt.imshow(tight)
        plt.subplot(132),plt.imshow(segmented_image)
        plt.subplot(133),plt.imshow(test1);
        plt.show()
        return contours,segmented_image
    else:
        return contours,segmented_image


def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]


def scaler_x(lst,z):
    xbig = 640
    for i,k in enumerate(z):
        factor = int(round((30/42)*(k)))
        xscale = (factor/xbig)
        for l in range(len(lst)):
            lst[l] = lst[l]*xscale
    return lst

def scaler_y(lst,z):
    ybig = 480
    for i,k in enumerate(z):
        factor = int(round((30/42)*(k)))

        yscale = (factor/ybig)
        
        for l in range(len(lst)):

            lst[l] = lst[l]*yscale
    return lst


def Reconstrcution(files,dir=r'models\classification_model\Tomogram_CNN_model_epoch=50.h5'):
    model = load_img_class_model(dir)
    z_plane,files = get_zplane(files,classi_model=model,infocus=True)
    num_of_obj,z,files = detect_obj(files,model)
    contours,_ = image_segmentation(files)
    
    print('Getting Corrdinatees')

    for obj in range(num_of_obj):
        k = 0
        globals()['x_obj{}'.format(obj)] = []
        globals()['y_obj{}'.format(obj)] = []

        while True:
            #try:
            #print(len(contours),obj)
            if k < len(contours[obj]):
                globals()['x_obj{}'.format(obj)].append((contours[obj][k][0][0]))
                globals()['y_obj{}'.format(obj)].append((contours[obj][k][0][1]))
                k = k+1
            # except IndexError:
            #     print(IndexError)
                # break
            else:
                break



        globals()['x_obj{}'.format(obj)] = scaler_x(globals()['x_obj{}'.format(obj)],z)
        globals()['y_obj{}'.format(obj)] = scaler_y(globals()['y_obj{}'.format(obj)],z)

    print('Done Getting Corrdinatees')
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    z_p = np.arange(min(z_plane),max(z_plane),0.3)
    ax.set_ylim(0,40)
    ax.set_xlim(0,40) 

    for obj in range(num_of_obj):
        for i in range(len(z_p)):
            ax.plot(globals()['x_obj{}'.format(obj)],globals()['y_obj{}'.format(obj)],z_p[i],c='gray');
    print('Finished Plotting')
    
    ax.set_xlabel('X-position')
    ax.set_ylabel('Y-position')
    ax.set_zlabel('Z-position')
    plt.show()
