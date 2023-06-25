

import nni
import nni.retiarii.nn.pytorch as nn
from nni.retiarii import model_wrapper
import nni.retiarii.strategy as strategy
from nni.retiarii.evaluator import FunctionalEvaluator
from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig

from common_import import *


####################################################################################################
# Multi-trial
####################################################################################################






### evaluator生成 ###
def evaluate_model(model_cls, architecture_params):
    global device, criterion, transform, train_dataloader, validation_dataloader, test_dataloader
    global epochs, trained_epochs, trial_dir, cp_period, cp_dir, es_flg, train_metrics, test_metrics

    # もし前回からの続きであれば重み読み込み
    if ('cp_path' in globals()) and (globals()['cp_path'] is not None):
        with open(trial_dir+'/architecture_params.json', 'r') as f:
            architecture_params = json.load(f)
        model = model_cls(architecture_params)
        model.to(device)
        model.load_state_dict(torch.load(globals()['cp_path'])['model_state_dict'])
    else:
        with open(trial_dir+'/architecture_params.json', 'w') as f:
            json.dump(architecture_params, f)
        model = model_cls()
        model.to(device)

    # Multi GPU使用宣言
    if str(device) == 'cuda':
        model = torch.nn.DataParallel(model)
        torch.backends.cudnn.benchmark = True
        print("Multi GPU OK.")
        print("\n\n")

    # 最適化関数
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    if ('cp_path' in globals()) and (globals()['cp_path'] is not None):
        optimizer.load_state_dict(torch.load(globals()['cp_path'])['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    callbacks = {}
    def on_epoch_end(**kwargs):
        # trained_epochsのインクリメント
        incrementTrainedEpochs(params_path=trial_dir+"/params.json")
        # history.jsonに保存
        saveHistory(kwargs['history_epoch'],history_path=trial_dir+"/history.json")
        # nniに保存
        nni.report_intermediate_result(kwargs['history_epoch']['val_accuracy'])
    callbacks['on_epoch_end'] = on_epoch_end
    def on_function_end(**kwargs):
        # nniに最終精度レポート
        nni.report_final_result(kwargs['last_history']['val_accuracy'])
    callbacks['on_function_end'] = on_function_end

    train_history = train(
        model,
        device,
        # train_dataloader,
        validation_dataloader,
        criterion,
        optimizer,
        epochs-trained_epochs,
        initial_epoch=trained_epochs,
        validation_dataloader=validation_dataloader,
        cp_step=cp_period,
        cp_path=cp_dir+"/cp_weights_{epoch:03d}-{accuracy:.4f}.pth",
        project_dir=trial_dir,
        es_flg=es_flg,
        metrics_dict=train_metrics,
        callbacks=callbacks
    )

    # for epoch_idx in range(initial_epoch,initial_epoch+epoch):

    #     print_log(f"Epoch: {epoch_idx+1:>3}/{initial_epoch+epoch}")

    #     # 学習
    #     train_history = train_one_epoch(model, device, train_dataloader, criterion, optimizer, callbacks, train_metrics)

    #     # 検証
    #     validation_history = validation_one_epoch(model, device, validation_dataloader, loss_fn, callbacks, train_metrics)
    #     nni.report_intermediate_result(validation_history['val_accuracy'])

    #     # モデルの重みの保存(check point)
    #     if (cp_period is not None) and ((epoch_idx+1) % cp_period == 0):
    #         torch.save(
    #             {
    #                 'epoch': epoch_idx+1,
    #                 'loss': train_history['loss'],
    #                 'model_state_dict': model.module.state_dict(),
    #                 'optimizer_state_dict': optimizer.state_dict(),
    #             },
    #             cp_path.format(
    #                 epoch=epoch_idx+1,
    #                 loss=train_history['loss'],
    #                 accuracy=train_history['accuracy'] if ('accuracy' in train_history) else None,
    #                 acc=train_history['accuracy'] if ('accuracy' in train_history) else None,
    #                 auc=train_history['auc'] if ('auc' in train_history) else None,
    #                 precision=train_history['precision'] if ('precision' in train_history) else None,
    #                 recall=train_history['recall'] if ('recall' in train_history) else None,
    #                 specificity=train_history['specificity'] if ('specificity' in train_history) else None,
    #                 f1=train_history['f1'] if ('f1' in train_history) else None,
    #                 val_loss=validation_history['val_loss'] if val_flg and ('val_loss' in validation_history) else None,
    #                 val_accuracy=validation_history['val_accuracy'] if val_flg and ('val_accuracy' in validation_history) else None,
    #                 val_acc=validation_history['val_accuracy'] if val_flg and ('val_accuracy' in validation_history) else None,
    #                 val_auc=validation_history['val_auc'] if val_flg and ('val_auc' in validation_history) else None,
    #                 val_precision=validation_history['val_precision'] if val_flg and ('val_precision' in validation_history) else None,
    #                 val_recall=validation_history['val_recall'] if val_flg and ('val_recall' in validation_history) else None,
    #                 val_specificity=validation_history['val_specificity'] if val_flg and ('val_specificity' in validation_history) else None,
    #                 val_f1=validation_history['val_f1'] if val_flg and ('val_f1' in validation_history) else None
    #             )
    #         )

    #     # trained_epochsのインクリメント
    #     incrementTrainedEpochs(params_path=model_dir+"/params.json")
    #     # history.jsonに保存
    #     saveHistory(kwargs['history_epoch'],history_path=model_dir+"/history.json")

    if ('cp_path' in globals()) and (globals()['cp_path'] is not None):
        globals()['cp_path'] = None


evaluator = FunctionalEvaluator(evaluate_model)


### 探索戦略取得 ###
strategyRandom = strategy.Random(dedup=True)
# strategyPolicyBasedRL = strategy.PolicyBasedRL()





@model_wrapper
class NNISampleNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):

        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        last_dim = 1 if num_classes==2 else num_classes

        self.conv1 = nn.LayerChoice([
            nn.Conv2d(in_channels,32,3,2,0,bias=False),
            SeparableConv2d(in_channels,32,3,2,0),
            # DepthwiseSeparableConv(in_channels, 32)
        ])
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.LayerChoice([
            nn.Conv2d(32,64,3,bias=False),
            SeparableConv2d(32,64,3),
            # DepthwiseSeparableConv(32, 64)
        ])
        self.bn2 = nn.BatchNorm2d(64)
        #do relu here

        self.block1=Block(64,128,2,2,start_with_relu=False,grow_first=True)
        self.block2=Block(128,256,2,2,start_with_relu=True,grow_first=True)
        self.block3=Block(256,728,2,2,start_with_relu=True,grow_first=True)

        self.block4=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block5=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block6=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block7=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block8=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block9=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block10=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block11=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block12=Block(728,1024,2,2,start_with_relu=True,grow_first=False)
        self.dropout1 = nn.Dropout(nn.ValueChoice([0.25, 0.5, 0.75]))

        self.conv3 = nn.LayerChoice([
            nn.Conv2d(1024,1536,3,1,0,bias=False),
            SeparableConv2d(1024,1536,3,1,0),
            # DepthwiseSeparableConv(1024, 1536)
        ])
        self.bn3 = nn.BatchNorm2d(1536)

        #do relu here
        self.feature1 = nn.ValueChoice([2048, 4096])
        self.conv4 = nn.LayerChoice([
            nn.Conv2d(1536,self.feature1,3,1,1,bias=False),
            SeparableConv2d(1536,self.feature1,3,1,1)
        ])
        self.bn4 = nn.BatchNorm2d(self.feature1)

        self.fc = nn.Linear(self.feature1, last_dim)
        self.dropout2 = nn.Dropout(nn.ValueChoice([0.25, 0.35, 0.45]))
        self.last_activation = nn.Softmax(dim=1) if num_classes>2 else nn.Sigmoid()
        
        initialize_parameters(self)

    def features(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)

        return x

    def logits(self, features):
        x = self.relu(features)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.dropout2(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x




