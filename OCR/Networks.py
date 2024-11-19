import torch
import random
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
from OCR.Modules import Easter2
from OCR.Losses import CustomCTC
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple



class CTCNetwork(ABC):
    def __init__(self, model: nn.Module, ctc_type: str = "default", architecture: str = "ocr_architecture", input_width: int = 2000, input_height: int = 80) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.architecture = architecture
        self.image_height = input_height
        self.image_width = input_width
        self.num_classes = 80
        self.model = model
        self.ctc_type = ctc_type
        self.criterion = nn.CTCLoss(blank=0, reduction="sum", zero_infinity=True) if self.ctc_type == "default" else CustomCTC()
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def full_train(self):
        for param in self.model.parameters():
            param.requires_grad = True

    @abstractmethod
    def get_input_shape(self) -> List[int]:
        raise NotImplementedError

    @abstractmethod
    def fine_tune(self, checkpoint_path: str):
        raise NotImplementedError
    
    @abstractmethod
    def forward(self, data: Tuple):
        raise NotImplementedError

    @abstractmethod
    def test(self, data: Tuple, all_data: bool) -> Tuple[List, List]:
        raise NotImplementedError
    
    def evaluate(self, data_loader, silent: bool):
        val_ctc_losses = []
        self.model.eval()

        for _, data in tqdm(enumerate(data_loader), total=len(data_loader), disable=silent):
            images, _, _ = [d.to(self.device) for d in data]
            with torch.no_grad():
                loss = self.forward(data)
                val_ctc_losses.append(loss / images.size(0))

        val_loss = torch.mean(torch.tensor(val_ctc_losses))

        return val_loss.item()

    def train(
        self,
        data_batch,
        clip_grads: bool = True,
        grad_clip: int = 5,
    ):
        self.model.train()

        loss = self.forward(data_batch)

        self.optimizer.zero_grad()
        loss.backward()

        if clip_grads:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
        self.optimizer.step()

        return loss.item()

    def get_checkpoint(self):
        checkpoint = {
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

        return checkpoint
    
    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def export_onnx(self, out_dir: str, model_name: str = "model", opset: int = 17) -> None:
        self.model.eval()

        model_input = torch.randn(self.get_input_shape(), device=self.device)
   
        """
        model_input = torch.randn(
            [1, 1, self.image_height, self.image_width], device=self.device
        )
        """
        out_file = f"{out_dir}/{model_name}.onnx"

        torch.onnx.export(
            self.model,
            model_input,
            out_file,
            export_params=True,
            opset_version=opset,
            verbose=False,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )

        print(f"Onnx file exported to: {out_file}")



class EasterNetwork(CTCNetwork):
    def __init__(
        self,
        image_width: int = 3200,
        image_height: int = 100,
        num_classes: int = 80,
        mean_pooling: bool = True,
        ctc_type: str = "default",
        ctc_reduction: str = "mean",
        learning_rate: float = 0.0005
    ) -> None:

        self.architecture = "OCR"
        self.image_width = image_width
        self.image_height = image_height
        self.num_classes = num_classes
        self.mean_pooling = mean_pooling
        self.device = "cuda"
        self.ctc_type = ctc_type
        self.ctc_reduction = "mean" if ctc_reduction == "mean" else "sum"
        self.learning_rate = learning_rate

        self.model = Easter2(
            input_width=self.image_width,
            input_height=self.image_height,
            vocab_size=self.num_classes,
            mean_pooling=self.mean_pooling
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )

        self.criterion = nn.CTCLoss(
            blank=0,
            reduction=self.ctc_reduction,
            zero_infinity=False) if self.ctc_type == "default" else CustomCTC()
        
        self.fine_tuning = False

        super().__init__(self.model, self.ctc_type, self.architecture, self.image_width, self.image_height)

        print(f"Network -> Architecture: {self.architecture}, input width: {self.image_width}, input height: {self.image_height}")

    def get_input_shape(self):
        return [self.num_classes, self.image_height, self.image_width]


    def fine_tune(self, checkpoint_path: str):
        self.load_checkpoint(checkpoint_path)
        
        trainable_layers = ["conv1d_5"]

        for param in self.model.named_parameters():
        
            for train_layers in trainable_layers:
                if train_layers not in param[0]:
                    param[1].data.requires_grad = False
                else:
                    if "easter" not in param[0]:
                        print(f"Unfreezing layer: {param[0]}")
                        param[1].data.requires_grad = True

        self.fine_tuning = True
        

    def load_model(self, checkpoint_path: str):
        self.load_checkpoint(checkpoint_path)

    def forward(self, data):
        images, targets, target_lengths = data
        images = torch.squeeze(images).to(self.device)
        targets = targets.to(self.device)
        target_lengths = target_lengths.to(self.device)

        logits = self.model(images)
        logits = logits.permute(2, 0, 1)
        log_probs = F.log_softmax(logits, dim=2)

        batch_size = images.size(0)
        input_lengths = torch.LongTensor(
            [logits.size(0)] * batch_size
        )  # i.e. time steps
        target_lengths = torch.flatten(target_lengths)

        loss = self.criterion(log_probs, targets, input_lengths, target_lengths)

        return loss
    

    def test(self, data, all_data: bool = False):
        self.model.eval()

        images, targets, target_lengths = data
        images = torch.squeeze(images).to(self.device)
        targets = targets.to(self.device)
        target_lengths = target_lengths.to(self.device)

        with torch.no_grad():
            logits = self.model(images)
        logits = logits.permute(0, 2, 1)
        logits = logits.cpu().detach().numpy()
        
        target_index = 0
        if not all_data:
            sample_idx = random.randint(0, logits.shape[0]-1)

            for b_idx, (logit, target_length) in enumerate(zip(logits, target_lengths)):
                if b_idx == sample_idx:
                    gt_label = targets[target_index:target_index+target_length+1]
                    gt_label = gt_label.cpu().detach().numpy().tolist()

                    return [logit], [gt_label]
                target_index += target_length

        else:
            gt_labels = []

            for _, (logit, target_length) in enumerate(zip(logits, target_lengths)):
                gt_label = targets[target_index:target_index+target_length+1]
                gt_label = gt_label.cpu().detach().numpy().tolist()
                gt_labels.append(gt_label)

                target_index += target_length

            return logits, gt_labels