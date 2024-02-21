from Models import ARCI as pr
from torch.utils.tensorboard import SummaryWriter
from pyhealth.trainer import Trainer
import torch.nn as nn
time_step = 4
import torch
import Baselines_MIMIC.BLUtils as ut


def initialize_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.xavier_normal_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0)
args = ut.ArgParser().parse_args()
dataset, train_loader, val_loader, test_loader = ut.get_loader(value=None,cross=args.index)

for tmp in [0.1]:
    for cl in [0.2]:
        for n_heads in [4]:
            for embed_dim in [256]:
                dir_tensorboard = f'Outputs_new_ZJ/MIMIC_RESULTS_Interpretability_mimic/dim_{args.dim}_heads_{n_heads}_CL_{str(cl).replace(".","_")}_CE_{str(args.ce).replace(".","_")}_tp_{str(tmp).replace(".","_")}_{args.index}'
                writer = SummaryWriter(log_dir=dir_tensorboard)
                model = pr.PresRec(
                    dataset=dataset,
                    feature_keys=["drugs_hist"],
                    label_key="drugs",
                    mode="multilabel",
                    embedding_dim_temp=embed_dim,
                    n_time_step=time_step,
                    n_heads=n_heads,
                    tmpr=tmp,
                    #interpretability=True,
                    cl=cl,
                    ce=args.ce,
                    device=args.cuda
                )
                model.to(f'cuda:{args.cuda}')
                trainer = Trainer(model=model)
                output = trainer.train(
                train_dataloader=train_loader,
                    max_grad_norm=1,
                    val_dataloader=val_loader,
                    epochs=20,
                    optimizer_class= torch.optim.Adam,
                    monitor="pr_auc_samples",
                )

                y_true, y_prob, loss = trainer.inference(test_loader)
                ut.evaluate(y_prob, y_true, writer, model)
                writer.close()