import time
from functools import partial

import lightning as L
from lightning import Fabric
from lightning.fabric.strategies import FSDPStrategy
import torch
import torch.nn.functional as F
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import torchmetrics
from torchvision import transforms
from torchvision.models import vit_l_16
from torchvision.models import ViT_L_16_Weights
from torchvision.models.vision_transformer import EncoderBlock
from watermark import watermark

from local_utilities import get_dataloaders_cifar10


def train(num_epochs, model, optimizer, train_loader, val_loader, fabric):

    for epoch in range(num_epochs):
        train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(fabric.device)

        model.train()
        for batch_idx, (features, targets) in enumerate(train_loader):
            model.train()
            
            ### FORWARD AND BACK PROP   
            logits = model(features)
            loss = F.cross_entropy(logits, targets)
            
            optimizer.zero_grad()
            fabric.backward(loss)

            ### UPDATE MODEL PARAMETERS
            optimizer.step()

            ### LOGGING
            if not batch_idx % 50:
                fabric.print(f"Epoch: {epoch+1:04d}/{num_epochs:04d} | Batch {batch_idx:04d}/{len(train_loader):04d} | Loss: {loss:.4f}")

            model.eval()
            with torch.no_grad():
                predicted_labels = torch.argmax(logits, 1)
                train_acc.update(predicted_labels, targets)

        ### MORE LOGGING
        model.eval()
        with torch.no_grad():
            val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(fabric.device)

            for (features, targets) in val_loader:
                outputs = model(features)
                predicted_labels = torch.argmax(outputs, 1)
                val_acc.update(predicted_labels, targets)

            fabric.print(f"Epoch: {epoch+1:04d}/{num_epochs:04d} | Train acc.: {train_acc.compute()*100:.2f}% | Val acc.: {val_acc.compute()*100:.2f}%")
            train_acc.reset(), val_acc.reset()


if __name__ == "__main__":

    auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={EncoderBlock})
    strategy = FSDPStrategy(auto_wrap_policy=auto_wrap_policy, activation_checkpointing=EncoderBlock)

    fabric = Fabric(accelerator="cuda", devices=4, strategy=strategy)
    fabric.launch()

    L.seed_everything(123)
    fabric.print(watermark(packages="torch,lightning", python=True))
    fabric.print("Torch CUDA available?", torch.cuda.is_available())


    ##########################
    ### 1 Loading the Dataset
    ##########################
    train_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                           #transforms.RandomCrop((224, 224)),
                                           transforms.ToTensor()])
    
    test_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                          #transforms.CenterCrop((224, 224)),
                                          transforms.ToTensor()])
    
    train_loader, val_loader, test_loader = get_dataloaders_cifar10(
        batch_size=64, 
        num_workers=1, 
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        validation_fraction=0.1)
    
    train_loader, val_loader, test_loader = fabric.setup_dataloaders(
        train_loader, val_loader, test_loader)


    #########################################
    ### 2 Initializing the Model
    #########################################

    model = vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_V1)

    # replace output layer
    model.heads.head = torch.nn.Linear(in_features=1024, out_features=10)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    model, optimizer = fabric.setup(model, optimizer)

    #########################################
    ### 3 Finetuning
    #########################################

    start = time.time()
    train(
        num_epochs=1,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        fabric=fabric
    )

    end = time.time()
    elapsed = end-start
    fabric.print(f"Time elapsed {elapsed/60:.2f} min")
    fabric.print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")

    #########################################
    ### 4 Evaluation
    #########################################
    
    with torch.no_grad():
        model.eval()
        test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(fabric.device)

        for (features, targets) in test_loader:
            outputs = model(features)
            predicted_labels = torch.argmax(outputs, 1)
            test_acc.update(predicted_labels, targets)

    fabric.print(f"Test accuracy {test_acc.compute()*100:.2f}%")