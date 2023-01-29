from common_import import *

print(f"START PROGRAM: {datetime.datetime.now()}")

###パラメータ###
# retrain_dir = "OriginalNet_20220731-130612_epoch50"
# retrain_dir = "OriginalNetNonDrop_20220731-131333_epoch50"
# retrain_dir = "OriginalNet_20221121-174136_epoch50"
retrain_dir = "Xception_20220812-065159_epoch15"
retrain_epochs = 0
gpu_count = 32


###パラメータ読込###
model_dir = "/home/toshi/fake_detection/model/"
params = loadParams(model_dir+retrain_dir+"/params.json")
model_structure = params['model_structure']
epochs = params['epochs']
batch_size_per_gpu = params['batch_size_per_gpu']
batch_size = batch_size_per_gpu * gpu_count
validation_rate = params['validation_rate']
test_rate = params['test_rate']
cp_period = params['cp_period']
# cp_period = 2
data_dir = params['data_dir']
classes = params['classes']
image_size = tuple(params['image_size'])
es_flg = params['es_flg']
rotation_range=params['rotation_range']
width_shift_range=params['width_shift_range']
height_shift_range=params['height_shift_range']
brightness_range=params['brightness_range']
shear_range=params['shear_range']
zoom_range=params['zoom_range']
channel_shift_range=params['channel_shift_range']
horizontal_flip=params['horizontal_flip']
vertical_flip=params['vertical_flip']


###トレーニングタイプ分類###
if 0 == len(glob.glob(model_dir+retrain_dir+'/checkpoint/*.h5')):
    print(retrain_dir+" has no checkpoint.")
    exit()
exist_learned_model = True if 0!=len(glob.glob(model_dir+retrain_dir+'/model.h5')) else False

newest_checkpoint_path = list(sorted(glob.glob(model_dir+retrain_dir+'/checkpoint/*.h5')))[-1]
# expect_epochs = int(re.findall(r'^.+epoch_(\d+).+?$', retrain_dir)[0])
expect_epochs = epochs
e = re.findall(r'^cp_model_(\d+)(\+\d+)?-.+?\.h5$', os.path.basename(newest_checkpoint_path))
if e[0][1]=='':
    saved_epochs = int(e[0][0])
else:
    saved_epochs = int(e[0][0]) + int(e[0][1])

print(newest_checkpoint_path)
print("expected epochs: ",end="")
print(expect_epochs)
print("saved epochs: ",end="")
print(saved_epochs)


###モデルの読み込み###
model = globals()['load'+model_structure](input_shape=image_size,weights_path=newest_checkpoint_path)
model.summary()


###Generator作成###
print(f"START CREATE GENERATOR: {datetime.datetime.now()}")
class_file_num = {}
class_weights = {}
train_data = []
validation_data = []
test_data = []
train_rate = 1 - validation_rate - test_rate
s1 = (int)(59*train_rate)
s2 = (int)(59*(train_rate+validation_rate))
id_list = list(range(62))
id_list.remove(13)
id_list.remove(14)
id_list.remove(18)
# random.shuffle(id_list)
train_id_list = id_list[ : s1]
validation_id_list = id_list[s1 : s2]
test_id_list = id_list[s2 : ]
print("\tTRAIN IMAGE DATA ID: ",end="")
print(train_id_list)
print("\tVALIDATION IMAGE DATA ID: ",end="")
print(validation_id_list)
print("\tTEST IMAGE DATA ID: ",end="")
print(test_id_list)
del id_list
data_num = 0
for l,c in enumerate(classes):
    image_path_list = sorted(glob.glob(data_dir+"/"+c+"/*"))
    path_num = len(image_path_list)
    data_num += path_num
    regexp = r'^.+?id(?P<id>(\d+))(_id(?P<id2>\d+))?_(?P<key>\d+)_(?P<num>\d+).(?P<ext>.{2,4})$'
    past_path = image_path_list[0]
    movie_image_list = []
    for i in range(1,len(image_path_list)):
        past_ids = re.search(regexp,past_path).groupdict()
        now_ids = re.search(regexp,image_path_list[i]).groupdict()
        if (past_ids['id']==now_ids['id']) and (past_ids['id2']==None or past_ids['id2']==now_ids['id2']) and (past_ids['key']==now_ids['key']):
            movie_image_list.append([image_path_list[i],l])
        else:
            if int(past_ids['id']) in train_id_list:
                train_data.append(movie_image_list)
            elif int(past_ids['id']) in validation_id_list:
                validation_data.append(movie_image_list)
            elif int(past_ids['id']) in test_id_list:
                test_data.append(movie_image_list)
            movie_image_list = []
            movie_image_list.append([image_path_list[i],l])
        past_path = image_path_list[i]
    # 不均衡データ調整
    class_file_num[c] = path_num
    if l==0:
        n = class_file_num[c]
    class_weights[l] = 1 / (class_file_num[c]/n)
print("\tMOVIE NUM: " + str(len(train_data)+len(validation_data)+len(test_data)))

train_data = list(chain.from_iterable(train_data))
validation_data = list(chain.from_iterable(validation_data))
test_data = list(chain.from_iterable(test_data))
train_data_num = len(train_data)
validation_data_num = len(validation_data)
test_data_num = len(test_data)
print("\tALL IMAGE DATA NUM: " + str(data_num))
print("\tTRAIN IMAGE DATA NUM: " + str(train_data_num))
print("\tVALIDATION IMAGE DATA NUM: " + str(validation_data_num))
print("\tTEST IMAGE DATA NUM: " + str(test_data_num))
def makeGenerator(data,subset="training"):
    return ImageIterator(
        data,
        batch_size=batch_size,
        target_size=image_size[:2],
        color_mode='rgb',
        seed=1,
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        brightness_range=brightness_range,
        shear_range=shear_range,
        zoom_range=zoom_range,
        channel_shift_range=channel_shift_range,
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip,
        rescale=1./255,
        data_format='channels_last',
        subset=subset)
train_generator = makeGenerator(train_data,"training")
validation_generator = makeGenerator(validation_data,"validation")
test_generator = makeGenerator(test_data,"test")
del train_data
del validation_data
del test_data
print(f"FINISH CREATE GENERATOR: {datetime.datetime.now()}")


###条件出力###
print("\tGPU COUNT: " + str(gpu_count))
print("\tBATCH SIZE: " + str(batch_size))
print("\tVALIDATION RATE: " + str(validation_rate))
print("\tTEST RATE: " + str(test_rate))
print("\tTRAIN DATA NUM: " + str(train_data_num))
print("\tVALIDATION DATA NUM: " + str(validation_data_num))
print("\tTEST DATA NUM: " + str(test_data_num))
print("\tCHECKPOINT PERIOD: " + str(cp_period))
print("\tDATA DIRECTORY: " + str(data_dir))
print("\tCLASSES: " + str(classes))
print("\tCLASSES NUM: " + str(class_file_num))
print("\tIMAGE SIZE: " + str(image_size))
print("\tEARLY STOPPING: " + str(es_flg))
print("\tROTATION RANGE: " + str(rotation_range))
print("\tWIDTH SHIFT RANGE: " + str(width_shift_range))
print("\tHEIGHT SHIFT RANGE: " + str(height_shift_range))
print("\tBRIGHTNESS RANGE: " + str(brightness_range))
print("\tSHEAR RANGE: " + str(shear_range))
print("\tZOOM RANGE: " + str(zoom_range))
print("\tCHANNEL SHIFT RANGE: " + str(channel_shift_range))
print("\tHORIZONTAL FLIP: " + str(horizontal_flip))
print("\tVERTICAL FLIP: " + str(vertical_flip))
print("")
print("\tTRAINING EPOCHS: " + str(saved_epochs) + "〜" + str(expect_epochs))
print("\tRETRAINING EPOCHS: " + str(retrain_epochs))
print("")



###ディレクトリパス###
model_dir = model_dir+retrain_dir
cp_dir = model_dir+'/checkpoint'











testModel(model,test_generator)
if sh_callback.history:
    makeGraph(sh_callback.history,model_dir)
























### jsonパラメータ読込 ###
def loadParams(filename="./params.json"):
    with open(filename, 'r') as f:
        params = json.load(f)
    return params


### モデルテスト(画像単位) ###
def testModel(model,test_generator):
    print(f"START TEST: {datetime.datetime.now()}")
    loss_and_metrics = model.evaluate_generator(
        test_generator
    )
    print("Test loss:",loss_and_metrics[0])
    print("Test accuracy:",loss_and_metrics[1])
    print("Test AUC:",loss_and_metrics[2])
    print("Test Precision:",loss_and_metrics[3])
    print("Test Recall:",loss_and_metrics[4])
    print("Test TP:",loss_and_metrics[5])
    print("Test TN:",loss_and_metrics[6])
    print("Test FP:",loss_and_metrics[7])
    print("Test FN:",loss_and_metrics[8])
    print(f"FINISH TEST: {datetime.datetime.now()}")
