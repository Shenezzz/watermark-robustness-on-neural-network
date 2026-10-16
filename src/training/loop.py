import torch
import torch.nn.functional as F
 
 
def train_one_epoch(model, loader, optimizer, device, extra_loss_fn=None):
    """
    extra_loss_fn: optional callable(model) -> scalar tensor, added to the
    cross-entropy loss before backward(). This is the hook watermark
    embedding will use later (e.g. Uchida's weight-regularization term)
    instead of copy-pasting this whole function.
    """
    model.train() # training mode to enable training-specific behavious such as dropout
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        if extra_loss_fn is not None:
            loss = loss + extra_loss_fn(model)
        loss.backward()
        optimizer.step()
 
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
        
    # Returning the loss in one training round as total_loss/total and acc as correct/total
    return total_loss / total, correct / total
 
 
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
 
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total