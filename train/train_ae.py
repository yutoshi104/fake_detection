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


###パラメータ###
model_structure = "AutoEncoder"
epochs = 2
gpu_count = 2
batch_size = 32 * gpu_count
validation_rate = 0.1
test_rate = 0.1
cp_period = 1
data_dir = '../data'
# classes = ['yuto']
classes = ['Celeb-real-image']
# classes = ['Celeb-real-image-face']
image_size = (480, 640, 3)
# image_size = (512, 512, 3)
es_flg = False


###data augmentation###
rotation_range=15.0#0.0
width_shift_range=0.15#0.0
height_shift_range=0.15#0.0
shear_range=0.0#0.0
zoom_range=0.1#0.0
horizontal_flip=True#False
vertical_flip=False#False



###ディレクトリ作成###
t = time.strftime("%Y%m%d-%H%M%S")
model_dir = f'../model/{model_structure}_{t}_epoch{epochs}'
os.makedirs(model_dir, exist_ok=True)
cp_dir = f'../model/{model_structure}_{t}_epoch{epochs}/checkpoint'
os.makedirs(cp_dir, exist_ok=True)



###モデルの生成###
model = globals()['load'+model_structure](input_shape=image_size,gpu_count=gpu_count)
model.summary()



###Generator作成###
file_num = sum(os.path.isfile(os.path.join(data_dir, name)) for name in os.listdir(data_dir))
datagen = ImageDataGenerator(
    rescale=1./255,
    data_format='channels_last',
    validation_split=validation_rate+test_rate,
    rotation_range=rotation_range,
    width_shift_range=width_shift_range,
    height_shift_range=height_shift_range,
    shear_range=shear_range,
    zoom_range=zoom_range,
    horizontal_flip=horizontal_flip,
    vertical_flip=vertical_flip
)
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=image_size[:2],
    batch_size=batch_size,
    class_mode='input',
    shuffle=True,
    seed=1,
    color_mode="rgb",
    classes=classes,
    subset = "training" 
)
vt_generator = datagen.flow_from_directory(
    data_dir,
    target_size=image_size[:2],
    batch_size=batch_size,
    class_mode='input',
    shuffle=True,
    seed=1,
    color_mode="rgb",
    classes=classes,
    subset = "validation"
)
vt_length = vt_generator.__len__()
test_generator = islice(vt_generator,0,(int)(vt_length*test_rate/(validation_rate+test_rate)))
validation_generator = islice(vt_generator,(int)(vt_length*test_rate/(validation_rate+test_rate)))


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
    filepath=cp_dir+"/cp_weight_{epoch:03d}-{accuracy:.2f}.hdf5",
    monitor='val_loss',
    mode='auto',
    save_best_only=True,
    save_weights_only=True,
    verbose=1,
    period=cp_period
)
cb_list.append(cp_callback)


###条件出力###
print("\tGPU COUNT: " + gpu_count)
print("\tBATCH SIZE: " + batch_size)
print("\tVALIDATION RATE: " + validation_rate)
print("\tTEST RATE: " + test_rate)
print("\tCHECKPOINT PERIOD: " + cp_period)
print("\tDATA DIRECTORY: " + data_dir)
print("\tCLASSES: " + classes)
print("\tIMAGE SIZE: " + image_size)
print("\tEARLY STOPPING: " + es_flg)
print("\tROTATION RANGE: " + rotation_range)
print("\tWIDTH SHIFT RANGE: " + width_shift_range)
print("\tHEIGHT SHIFT RANGE: " + height_shift_range)
print("\tSHEAR RANGE: " + shear_range)
print("\tZOOM RANGE: " + zoom_range)
print("\tHORIZONTAL FLIP: " + horizontal_flip)
print("\tVERTICAL FLIP: " + vertical_flip)
print("")


###学習###
history = model.fit_generator(
    train_generator,
    validation_data=validation_generator,
    # steps_per_epoch=10,
    epochs=epochs,
    verbose=1,
    workers=8,
    use_multiprocessing=False,
    callbacks=cb_list
)



###テスト###
print("テスト中")
loss_and_metrics = model.evaluate_generator(
    test_generator
)
print("Test loss:",loss_and_metrics[0])
print("Test accuracy:",loss_and_metrics[1])



###モデルの保存###
print("Save model...")
model.save(f'{model_dir}/model.h5')
model.save_weights(f'{model_dir}/weight.hdf5')



###グラフ化###
fig = plt.figure()
plt.plot(range(1, len(history.history['accuracy'])+1), history.history['accuracy'], "-o")
plt.plot(range(1, len(history.history['val_accuracy'])+1), history.history['val_accuracy'], "-o")
plt.title('Model accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.grid()
plt.legend(['accuracy','val_accuracy'], loc='best')
fig.savefig(model_dir+"/result.png")
# plt.show()

print("Finish")