
####################################################################################################
###実行コマンド###
# nohup python train_simple.py > train_simple_`date +%Y%m%d_%H%M%S`.log &
###強制終了コマンド###
# jobs      (実行中のジョブ一覧)
# kill 1    (1のジョブを強制終了)
# fg 1      (1をフォアグランド実行に戻す)
# ctrl+c    (強制終了)
    # ctrl+z    (停止)
    # bg %1     (1をバックグラウンド実行で再開)
####################################################################################################


from common_import import *
# pprint([s for s in dir(torchvision.models) if s[0].isupper()], compact=True)
# exit()

#---------------------------------------------------------------------------------------------------
# ハイパーパラメータなどの設定値

structure_name = 'XceptionNet'
# structure_name = 'Vgg16'
# structure_name = 'InceptionV3'
epochs = 50
gpu_count = GPU_COUNT
# batch_size_per_gpu = 32 # 12GB/GPUの時 (これで8GBずつくらい)
batch_size_per_gpu = 128 # 32GB/GPUの時 (これで27GBずつくらい)
batch_size = batch_size_per_gpu * gpu_count
validation_rate = 0.1
test_rate = 0.1
cp_period = 1
data_dir = os.getenv('FAKE_DATA_PATH')
classes = ['Celeb-real-image-face-90', 'Celeb-synthesis-image-face-90']
image_size = (3, 256, 256)
data_shuffle = True
es_flg = False

# data augmentation
# rotation_range=15.0
# width_shift_range=0.15
# height_shift_range=0.15
# brightness_range = 0.0
# shear_range=0.0
# zoom_range=0.1
# horizontal_flip=True
# vertical_flip=False
rotation_range=0.0
width_shift_range=0.0
height_shift_range=0.0
brightness_range = 0.0
shear_range=0.0
zoom_range=0.0
horizontal_flip=False
vertical_flip=False


projects_dir = "./projects_simple/"















#---------------------------------------------------------------------------------------------------

# 再開
####################################################################################################
# train_simple.py [porject名]
####################################################################################################
if len(sys.argv) >= 2 and os.path.isdir(projects_dir+sys.argv[1]):
    print("This is retraining.\n")
    first_training = False

    retrain_dir = sys.argv[1]
    model_dir = f'{projects_dir}{retrain_dir}'
    cp_dir = f'{projects_dir}{retrain_dir}/checkpoint'

    if 0 == len(glob.glob(cp_dir+'/*.pth')):
        print("'"+retrain_dir+"' has no checkpoint.")
        exit()

    params = loadParams(model_dir+"/params.json")
    structure_name = params['structure_name']
    epochs = params['epochs']
    # trained_epochs = params['trained_epochs']
    batch_size_per_gpu = params['batch_size_per_gpu']
    batch_size = batch_size_per_gpu * gpu_count
    validation_rate = params['validation_rate']
    test_rate = params['test_rate']
    cp_period = params['cp_period']
    data_dir = params['data_dir']
    classes = params['classes']
    image_size = tuple(params['image_size'])
    data_shuffle = params['data_shuffle']
    es_flg = params['es_flg']
    rotation_range=params['rotation_range']
    width_shift_range=params['width_shift_range']
    height_shift_range=params['height_shift_range']
    brightness_range=params['brightness_range']
    shear_range=params['shear_range']
    zoom_range=params['zoom_range']
    horizontal_flip=params['horizontal_flip']
    vertical_flip=params['vertical_flip']

    cp_path = list(sorted(glob.glob(cp_dir+'/*.pth')))[-1]
    print("check point path: "+cp_path)

    # trained_epochsの記載と保存の差異合わせ
    trained_epochs = torch.load(cp_path)['epoch']
    params['trained_epochs'] = trained_epochs
    saveParams(params,filename=model_dir+"/params.json")
    # history.jsonの差異合わせ
    adjustHistory(trained_epochs, history_path=model_dir+"/history.json")

# 初回
else:
    print("This is first training.\n")
    first_training = True
    trained_epochs = 0

    t = now(format_str="%Y%m%d-%H%M%S")
    model_dir = f'{projects_dir}{structure_name}_{t}_epoch{epochs}'
    cp_dir = f'{projects_dir}{structure_name}_{t}_epoch{epochs}/checkpoint'



#---------------------------------------------------------------------------------------------------
# ニューラルネットワークの生成
print(f"START CREATE NETWORK: {now()}")
model = globals()[structure_name](image_size[0],len(classes))
if not first_training:
    model.load_state_dict(torch.load(cp_path)['model_state_dict'])
torchsummary.summary(model.to('cpu'), image_size, device='cpu')
model = model.to(device)
print(f"FINISH CREATE NETWORK: {now()}")
print("\n\n")


#---------------------------------------------------------------------------------------------------
# Multi GPU使用宣言
if str(device) == 'cuda':
    model = torch.nn.DataParallel(model)
    torch.backends.cudnn.benchmark = True
    print("Multi GPU OK.")
    print("\n\n")



#---------------------------------------------------------------------------------------------------
# 変換方法の指定

transform = getTransforms(
    rotation_range=rotation_range,
    width_shift_range=width_shift_range,
    height_shift_range=height_shift_range,
    brightness_range=brightness_range,
    shear_range=shear_range,
    zoom_range=zoom_range,
    horizontal_flip=horizontal_flip,
    vertical_flip=vertical_flip
)


# #---------------------------------------------------------------------------------------------------
# # テスト領域
# disp_dataloader = getMiniCelebDataLoader(
#     data_dir=data_dir,
#     classes=classes,
#     validation_rate=validation_rate,
#     test_rate=test_rate
# )
# dispImages(
#     data_loader=disp_dataloader,
#     file_path='aaa.jpg',
#     model=model,
#     classes=classes
# )
# exit()




#---------------------------------------------------------------------------------------------------
# 学習用／評価用のデータセットの作成

# CelebData取得
print(f"START CREATE DATASET: {now()}") 
train_dataloader, validation_dataloader, test_dataloader, \
data_num, class_file_num, class_weights = getCelebDataLoader(
    batch_size=batch_size,
    transform=transform,
    data_dir=data_dir,
    classes=classes,
    image_size=image_size,
    validation_rate=validation_rate,
    test_rate=test_rate,
    shuffle=data_shuffle
)
print(f"FINISH CREATE DATASET: {now()}")
print("\n\n")



#---------------------------------------------------------------------------------------------------
# 損失関数の設定
if len(classes)==2:
    criterion = nn.BCELoss()
else:
    criterion = nn.CrossEntropyLoss()


#---------------------------------------------------------------------------------------------------
# 最適化手法の設定
if first_training:
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01) 
else:
    optimizer = torch.load(cp_path)['optimizer_state_dict']

#---------------------------------------------------------------------------------------------------
# Metrics取得
train_metrics = getMetrics(device, mode="all", num_classes=len(classes), average='none')
test_metrics = getMetrics(device, mode="all", num_classes=len(classes), average='none')


#---------------------------------------------------------------------------------------------------
# ディレクトリ作成
if first_training:
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(cp_dir, exist_ok=True)
    print(f"CREATE DIRECTORY: '{structure_name}_{t}_epoch{epochs}'")


#---------------------------------------------------------------------------------------------------
# パラメータ保存
if first_training:
    params = {}
    params["structure_name"] = structure_name
    params["epochs"] = epochs
    params["trained_epochs"] = 0
    params["batch_size_per_gpu"] = batch_size_per_gpu
    params["validation_rate"] = validation_rate
    params["test_rate"] = test_rate
    params["cp_period"] = cp_period
    params["data_dir"] = data_dir
    params["classes"] = classes
    params["image_size"] = image_size
    params["data_shuffle"] = data_shuffle
    params["es_flg"] = es_flg
    params["rotation_range"] = rotation_range
    params["width_shift_range"] = width_shift_range
    params["height_shift_range"] = height_shift_range
    params["brightness_range"] = brightness_range
    params["shear_range"] = shear_range
    params["zoom_range"] = zoom_range
    params["horizontal_flip"] = horizontal_flip
    params["vertical_flip"] = vertical_flip
    saveParams(params,filename=model_dir+"/params.json")


#---------------------------------------------------------------------------------------------------
# 条件出力
print("\tEPOCHS: " + str(epochs))
print("\tFIRST TRAINING: " + str(first_training))
if not first_training:
    print("\tTRAINED EPOCHS: " + str(trained_epochs))
print("\tGPU COUNT: " + str(gpu_count))
print("\tBATCH SIZE: " + str(batch_size))
print("\tVALIDATION RATE: " + str(validation_rate))
print("\tTEST RATE: " + str(test_rate))
print("\tTRAIN DATA NUM: " + str(len(train_dataloader.dataset)))
print("\tVALIDATION DATA NUM: " + str(len(validation_dataloader.dataset)))
print("\tTEST DATA NUM: " + str(len(test_dataloader.dataset)))
print("\tCHECKPOINT PERIOD: " + str(cp_period))
print("\tDATA DIRECTORY: " + str(data_dir))
print("\tCLASSES: " + str(classes))
print("\tCLASSES NUM: " + str(class_file_num))
print("\tCLASSES WEIGHTS: " + str(class_weights))
print("\tIMAGE SIZE: " + str(image_size))
print("\tDATA SHUFFLE: " + str(data_shuffle))
print("\tEARLY STOPPING: " + str(es_flg))
print("\tROTATION RANGE: " + str(rotation_range))
print("\tWIDTH SHIFT RANGE: " + str(width_shift_range))
print("\tHEIGHT SHIFT RANGE: " + str(height_shift_range))
print("\tBRIGHTNESS RANGE: " + str(brightness_range))
print("\tSHEAR RANGE: " + str(shear_range))
print("\tZOOM RANGE: " + str(zoom_range))
print("\tHORIZONTAL FLIP: " + str(horizontal_flip))
print("\tVERTICAL FLIP: " + str(vertical_flip))
print("\n\n")


#---------------------------------------------------------------------------------------------------
# callbacks
callbacks = {}
def on_epoch_end(**kwargs):
    # trained_epochsのインクリメント
    incrementTrainedEpochs(params_path=model_dir+"/params.json")
    # history.jsonに保存
    saveHistory(kwargs['history_epoch'],history_path=model_dir+"/history.json")
callbacks['on_epoch_end'] = on_epoch_end


#---------------------------------------------------------------------------------------------------
# 学習
print(f"START TRAINING: {now()}")
train_history = train_epoch(
    model,
    device,
    train_dataloader,
    # validation_dataloader,
    criterion,
    optimizer,
    epochs-trained_epochs,
    initial_epoch=trained_epochs,
    validation_dataloader=validation_dataloader,
    save_step=cp_period,
    save_path=cp_dir+"/cp_weights_{epoch:03d}-{accuracy:.4f}.pth",
    project_dir=model_dir,
    es_flg=es_flg,
    metrics_dict=train_metrics,
    callbacks=callbacks
)
print("Train Result: ")
pprint(train_history)
print(f"FINISH TRAINING: {now()}")
print("\n\n")


#---------------------------------------------------------------------------------------------------
# モデル構造保存
torch.save(model.state_dict(), model_dir+'/model_weights.pth')


#---------------------------------------------------------------------------------------------------
# 評価
print(f"START TEST: {now()}")
test_history = test_epoch(
    model,
    device,
    test_dataloader,
    criterion,
    metrics_dict=test_metrics
)
print("Test Result: ")
pprint(test_history)

print(f"FINISH TEST: {now()}")
print("\n\n")


#---------------------------------------------------------------------------------------------------
# 結果グラフ描画
print(f"START DRAW RESULT: {now()}")

all_history = loadParams(filename=model_dir+'/history.json')

# loss描画
graph_data_loss = {'loss':all_history['loss']}
if 'val_loss' in all_history:
    graph_data_loss['val_loss'] = all_history['val_loss']
saveLossGraph(graph_data_loss, save_path=model_dir+'/loss.png', title='Model Loss (of this training)')

# accuracy描画
graph_data_acc = {}
if 'accuracy' in all_history:
    graph_data_acc['accuracy'] = all_history['accuracy']
if 'val_accuracy' in all_history:
    graph_data_acc['val_accuracy'] = all_history['val_accuracy']
if graph_data_acc != {}:
    saveLossGraph(graph_data_acc, save_path=model_dir+'/accuracy.png', title='Model Accuracy (of this training)')

# ROC曲線描画
if 'roc' in test_history:
    saveRocCurve(test_history['roc'], save_path=model_dir+'/roc.png', title='ROC Curve (of this training)')

print(f"FINISH DRAW RESULT: {now()}")
print("\n\n")
