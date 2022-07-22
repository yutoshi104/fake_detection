####################################################################################################
###実行コマンド###
# nohup python train.py > train_out.log &
###強制終了コマンド###
# jobs      (実行中のジョブ一覧)
# kill 1    (1のジョブを強制終了)
# fg 1      (1をフォアグランド実行に戻す)
# ctrl+c    (強制終了)
    # ctrl+z    (停止)
    # bg %1     (1をバックグラウンド実行で再開)
####################################################################################################

from common_import import *

print(f"START PROGRAM: {datetime.datetime.now()}")

###パラメータ###
retrain_dir = "Xception_20220720-133534_epoch30"
retrain_epochs = 10
gpu_count = 8



###パラメータ読込###
model_dir = "/home/toshi/fake_detection/model/"
params = loadParams(model_dir+retrain_dir+"/params.json")
epochs = params['epochs']
batch_size_per_gpu = params['batch_size_per_gpu']
batch_size = batch_size_per_gpu * gpu_count
validation_rate = params['validation_rate']
test_rate = params['test_rate']
cp_period = params['cp_period']
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

print("params: ",end="")
print(params)

###トレーニングタイプ分類###
# checkpointが存在しない場合は処理を終了
# 期待されているエポックの学習が終わっていなかったらその学習を行う(train_flg)
# 再学習エポックが指定されていたら再学習を行う(retrain_flg)
if 0 == len(glob.glob(model_dir+retrain_dir+'/checkpoint/*.h5')):
    print(retrain_dir+" has no checkpoint.")
    exit()
exist_learned_model = True if 0!=len(glob.glob(model_dir+retrain_dir+'/model.h5')) else False

newest_checkpoint_path = list(sorted(glob.glob(model_dir+retrain_dir+'/checkpoint/*.h5')))[-1]
# expect_epochs = int(re.findall(r'^.+epoch_(\d+).+?$', retrain_dir)[0])
expect_epochs = epochs
saved_epochs = int(re.findall(r'^cp_model_(\d+)-.+?\.h5$', os.path.basename(newest_checkpoint_path))[0])

print(newest_checkpoint_path)
print("expected epochs: ",end="")
print(expect_epochs)
print("saved epochs: ",end="")
print(saved_epochs)

train_flg = True if (expect_epochs > saved_epochs and not exist_learned_model) else False
retrian_flg = True if (retrain_epochs > 0) else False



###モデルの読み込み###
model = models.load_model(newest_checkpoint_path)
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


###終了時の処理###
def test_and_save_and_graph():
    global model
    global model_dir
    global history
    global test_generator
    testModel(model,test_generator)
    saveModel(model,model_dir)
    makeGraph(history.history,model_dir)



###中断した学習の再開###
if train_flg:

    ###ディレクトリパス###
    model_dir = model_dir+retrain_dir
    cp_dir = model_dir+'/checkpoint'


    ###callback作成###
    cb_list = []
    if es_flg:
        ##↓監視する値の変化が停止した時に訓練を終了##
        es_callback = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=2,
            verbose=1,
            mode='auto'
        )
        cb_list.append(es_callback)
    cp_callback = callbacks.ModelCheckpoint(
        filepath=cp_dir+"/cp_model_{epoch+"+str(saved_epochs)+":03d}-{accuracy:.2f}.h5",
        monitor='val_loss',
        mode='auto',
        save_best_only=False,
        save_weights_only=False,
        verbose=1,
        period=cp_period
    )
    cb_list.append(cp_callback)
    sh_callback = saveHistory(filepath=model_dir+"/history.json")
    cb_list.append(sh_callback)


    ##サンプルデータセット使用###
    # train_generator,validation_generator,test_generator = getSampleData()


    ###学習###
    print(f"START TRAINING: {datetime.datetime.now()}")
    signal.signal(signal.SIGINT, test_and_save_and_graph)
    try:
        history = model.fit_generator(
            train_generator,
            validation_data=validation_generator,
            # steps_per_epoch=10,
            epochs=expect_epochs-saved_epochs,
            class_weight=class_weights,
            verbose=1,
            workers=8,
            use_multiprocessing=False,
            callbacks=cb_list
        )
    except Exception as e:
        print(e)
    print(f"FINISH TRAINING: {datetime.datetime.now()}")

    testModel(model,test_generator)
    saveModel(model,model_dir)
    makeGraph(sh_callback.history,model_dir)

    print(f"FINISH PROGRAM: {datetime.datetime.now()}")






###再学習###
if retrain_flg:

    ###ディレクトリ作成###
    t = time.strftime("%Y%m%d-%H%M%S")
    model_dir = f'../model/{model_structure+"_retrain"}_{t}_epoch{expect_epochs+retrain_epochs}'
    os.makedirs(model_dir, exist_ok=True)
    cp_dir = model_dir+'/checkpoint'
    os.makedirs(cp_dir, exist_ok=True)

    ###パラメータ保存###
    params['epoch'] = expect_epochs + retrain_epochs
    saveParams(params,filename=model_dir+"/params.json")

    ###callback作成###
    cb_list = []
    if es_flg:
        ##↓監視する値の変化が停止した時に訓練を終了##
        es_callback = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=2,
            verbose=1,
            mode='auto'
        )
        cb_list.append(es_callback)
    cp_callback = callbacks.ModelCheckpoint(
        filepath=cp_dir+"/cp_model_{epoch+"+str(expect_epochs)+":03d}-{accuracy:.2f}.h5",
        monitor='val_loss',
        mode='auto',
        save_best_only=False,
        save_weights_only=False,
        verbose=1,
        period=cp_period
    )
    cb_list.append(cp_callback)
    sh_callback = saveHistory(filepath=model_dir+"/history.json")
    cb_list.append(sh_callback)


    ##サンプルデータセット使用###
    # train_generator,validation_generator,test_generator = getSampleData()


    ###学習###
    print(f"START TRAINING: {datetime.datetime.now()}")
    signal.signal(signal.SIGINT, test_and_save_and_graph)
    try:
        history = model.fit_generator(
            train_generator,
            validation_data=validation_generator,
            # steps_per_epoch=10,
            epochs=retrain_epochs,
            class_weight=class_weights,
            verbose=1,
            workers=8,
            use_multiprocessing=False,
            callbacks=cb_list
        )
    except Exception as e:
        print(e)
    print(f"FINISH TRAINING: {datetime.datetime.now()}")

    testModel(model,test_generator)
    saveModel(model,model_dir)
    makeGraph(sh_callback.history,model_dir)

    print(f"FINISH PROGRAM: {datetime.datetime.now()}")
