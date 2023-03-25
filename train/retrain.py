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
# retrain_dir = "OriginalNet_20220731-130612_epoch50"
# retrain_dir = "OriginalNetNonDrop_20220731-131333_epoch50"
# retrain_dir = "OriginalNet_20221121-174136_epoch50"
retrain_dir = "Xception_20230112-045215_epoch100"
retrain_epochs = 0
gpu_count = 16


###コマンドライン引数###
if 2 <= len(sys.argv):
    retrain_dir = sys.argv[1]
if 3 <= len(sys.argv):
    gpu_count = int(sys.argv[2])

print("\tRETRAIN DIR: " + str(retrain_dir))
print("\tGPU COUNT: " + str(gpu_count))


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
edge=params['edge'] if ('edge' in params) else None

print("params: ",end="")
print(params)

###トレーニングタイプ分類###
# checkpointが存在しない場合は処理を終了
# 期待されているエポックの学習が終わっていなかったらその学習を行う(train_flg)
# 再学習エポックが指定されていたら再学習を行う
if 0 == len(glob.glob(model_dir+retrain_dir+'/checkpoint/*.h5')):
    print(retrain_dir+" has no checkpoint.")
    exit()
exist_learned_model = True if 0!=len(glob.glob(model_dir+retrain_dir+'/model.h5')) else False

newest_checkpoint_path = list(sorted(glob.glob(model_dir+retrain_dir+'/checkpoint/*.h5')))[-1]
# expect_epochs = int(re.findall(r'^.+epoch_(\d+).+?$', retrain_dir)[0])
expect_epochs = epochs
e = re.findall(r'^cp_model_(\d+)(\+\d+)?-.+?\.h5$', os.path.basename(newest_checkpoint_path))
print(e[0])
if e[0][1]=='':
    saved_epochs = int(e[0][0])
else:
    saved_epochs = int(e[0][0]) + int(e[0][1])

print(newest_checkpoint_path)
print("expected epochs: ",end="")
print(expect_epochs)
print("saved epochs: ",end="")
print(saved_epochs)

train_flg = True if (expect_epochs > saved_epochs or not exist_learned_model) else False
retrian_flg = True if (retrain_epochs > 0) else False



###モデルの読み込み###
model = globals()['load'+model_structure](input_shape=image_size,weights_path=newest_checkpoint_path)
model.summary()


###Generator作成###
print(f"START CREATE GENERATOR: {datetime.datetime.now()}")
train_generator, validation_generator, test_generator, class_file_num, class_weights = makeImageDataGenerator_Celeb(
    data_dir,
    classes,
    validation_rate,
    test_rate,
    batch_size,
    image_size,
    rotation_range,
    width_shift_range,
    height_shift_range,
    brightness_range,
    shear_range,
    zoom_range,
    channel_shift_range,
    horizontal_flip,
    vertical_flip,
    edge
)
print(f"FINISH CREATE GENERATOR: {datetime.datetime.now()}")


###条件出力###
print("\tGPU COUNT: " + str(gpu_count))
print("\tBATCH SIZE: " + str(batch_size))
print("\tVALIDATION RATE: " + str(validation_rate))
print("\tTEST RATE: " + str(test_rate))
print("\tTRAIN DATA NUM: " + str(train_generator.len()))
print("\tVALIDATION DATA NUM: " + str(validation_generator.len()))
print("\tTEST DATA NUM: " + str(test_generator.len()))
print("\tCHECKPOINT PERIOD: " + str(cp_period))
print("\tDATA DIRECTORY: " + str(data_dir))
print("\tCLASSES: " + str(classes))
print("\tCLASSES NUM: " + str(class_file_num))
print("\tCLASSES WEIGHTS: " + str(class_weights))
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
print("\tEDGE: " + str(edge))
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


###ディレクトリパス###
model_dir = model_dir+retrain_dir
cp_dir = model_dir+'/checkpoint'


###中断した学習の再開###
if train_flg:

    ###history修正###
    his = loadParams(model_dir+"/history.json")
    for key in his.keys():
        his[key] = his[key][:saved_epochs]
    saveParams(his,model_dir+"/history.json")


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
        filepath=cp_dir+"/cp_model_"+str(saved_epochs).zfill(3)+"+{epoch:03d}-{accuracy:.2f}.h5",
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








###再学習###
print(retrain_epochs)
if retrain_epochs > 0:

    ###ディレクトリ作成###
    past_model_dir = model_dir
    t = time.strftime("%Y%m%d-%H%M%S")
    model_dir = f'../model/{model_structure+"_retrain"}_{t}_epoch{expect_epochs+retrain_epochs}'
    os.makedirs(model_dir, exist_ok=True)
    cp_dir = model_dir+'/checkpoint'
    os.makedirs(cp_dir, exist_ok=True)

    ###パラメータ保存###
    params['epochs'] = expect_epochs+retrain_epochs
    params['total_epoch'] = (params['total_epoch'] if 'total_epoch' in params else expect_epochs) + retrain_epochs
    params['past_model_dir'] = past_model_dir
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
        filepath=cp_dir+"/cp_model_"+str(expect_epochs)+"+{epoch:03d}-{accuracy:.2f}.h5",
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
if sh_callback.history:
    makeGraph(sh_callback.history,model_dir)

print(f"FINISH PROGRAM: {datetime.datetime.now()}")
