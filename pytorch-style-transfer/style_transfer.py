import torch
from torch import nn
import torchvision
from torchvision import transforms
from torchvision.models import vgg19, VGG19_Weights
from PIL import Image
import os
import matplotlib.pyplot as plt

# ─────────────── setting ───────────────
rgb_mean = torch.tensor([0.485, 0.456, 0.406])
rgb_std = torch.tensor([0.229, 0.224, 0.225])
content_layers = [25]
style_layers = [0, 5, 10, 19, 28]

# ─────────────── helper function ───────────────
def preprocess(img, image_shape):
    img = img.convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(image_shape),
        transforms.ToTensor(),
        transforms.Normalize(mean=rgb_mean, std=rgb_std)
    ])
    return transform(img).unsqueeze(0)

def postprocess(img):
    img = img[0].cpu().detach()
    img = torch.clamp(img.permute(1, 2, 0) * rgb_std + rgb_mean, 0, 1)
    return transforms.ToPILImage()(img.permute(2, 0, 1))

def gram(X):
    if X.dim() == 4:
        _, c, h, w = X.shape
        X = X.view(c, -1)
    elif X.dim() == 3:
        c, h, w = X.shape
        X = X.view(c, -1)
    else:
        raise ValueError(f"Invalid tensor shape for gram matrix: {X.shape}")
    return torch.matmul(X, X.T) / (c * h * w)

# ─────────────── loss function ───────────────
def content_loss(Y_hat, Y):
    return torch.mean((Y_hat - Y.detach()) ** 2)

def style_loss(Y_hat, gram_Y):
    return torch.mean((gram(Y_hat) - gram_Y.detach()) ** 2)

def tv_loss(Y_hat):
    return 0.5 * (
        torch.mean(torch.abs(Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1])) +
        torch.mean(torch.abs(Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]))
    )

def compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram,
                 content_weight, style_weight, tv_weight):
    contents_l = [content_loss(Y_hat, Y) * content_weight for Y_hat, Y in zip(contents_Y_hat, contents_Y)]
    styles_l = [style_loss(Y_hat, gram_Y) * style_weight for Y_hat, gram_Y in zip(styles_Y_hat, styles_Y_gram)]
    tv_l = tv_loss(X) * tv_weight
    return sum(contents_l) + sum(styles_l) + tv_l

# ─────────────── Composite Image Class ───────────────
class SynthesizedImage(nn.Module):
    def __init__(self, img_shape):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(img_shape))

    def forward(self):
        return self.weight

def extract_features(X, net, content_layers, style_layers):
    contents = []
    styles = []
    for i in range(len(net)):
        X = net[i](X)
        if i in style_layers:
            styles.append(X)
        if i in content_layers:
            contents.append(X)
    return contents, styles

# ─────────────── Default Style Transfer ───────────────
def run_style_transfer(content_path, style_path, image_shape=(300, 450),
                       content_weight=5, style_weight=500, tv_weight=10,
                       lr=0.05, num_epochs=1000, lr_decay_epoch=100):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    content_img = Image.open(content_path)
    style_img = Image.open(style_path)
    content_X = preprocess(content_img, image_shape).to(device)
    style_X = preprocess(style_img, image_shape).to(device)

    vgg = vgg19(weights=VGG19_Weights.DEFAULT).features.to(device).eval()
    max_required_layer = max(content_layers + style_layers)
    net = nn.Sequential(*[vgg[i] for i in range(max_required_layer + 1)]).to(device)
    for param in net.parameters():
        param.requires_grad = False

    contents_Y, _ = extract_features(content_X, net, content_layers, style_layers)
    _, styles_Y = extract_features(style_X, net, content_layers, style_layers)
    styles_Y_gram = [gram(y) for y in styles_Y]

    gen_img = SynthesizedImage(content_X.shape).to(device)
    gen_img.weight.data.copy_(content_X.data)
    optimizer = torch.optim.Adam(gen_img.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_decay_epoch, 0.8)

    os.makedirs("results", exist_ok=True)
    losses = []
    fig, ax = plt.subplots()
    plt.ion()

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        X = gen_img()
        contents_Y_hat, styles_Y_hat = extract_features(X, net, content_layers, style_layers)
        loss = compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram,
                            content_weight, style_weight, tv_weight)
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}, Loss {loss.item():.2f}")
            ax.clear()
            ax.plot(losses, label="Loss")
            ax.set_title("Style Transfer Loss Curve")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()
            plt.pause(0.001)

    plt.ioff()
    result = postprocess(gen_img())
    content_name = os.path.basename(content_path).split('.')[0]
    style_name = os.path.basename(style_path).split('.')[0]
    result_img_path = f"results/result_{content_name}_{style_name}.png"
    result_graph_path = f"results/result_{content_name}_{style_name}_loss.png"
    result.save(result_img_path)
    plt.savefig(result_graph_path)
    print(f"Saved: {result_img_path}")
    print(f"Saved: {result_graph_path}")

# ─────────────── Style Highlighted Version ───────────────
def run_style_transfer_strong_style(content_path, style_path, image_shape=(300, 450),
                       content_weight=1, style_weight=3000, tv_weight=5,
                       lr=0.03, num_epochs=1000, lr_decay_epoch=100):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    content_img = Image.open(content_path)
    style_img = Image.open(style_path)
    content_X = preprocess(content_img, image_shape).to(device)
    style_X = preprocess(style_img, image_shape).to(device)

    vgg = vgg19(weights=VGG19_Weights.DEFAULT).features.to(device).eval()
    max_required_layer = max(content_layers + style_layers)
    net = nn.Sequential(*[vgg[i] for i in range(max_required_layer + 1)]).to(device)
    for param in net.parameters():
        param.requires_grad = False

    contents_Y, _ = extract_features(content_X, net, content_layers, style_layers)
    _, styles_Y = extract_features(style_X, net, content_layers, style_layers)
    styles_Y_gram = [gram(y) for y in styles_Y]

    gen_img = SynthesizedImage(content_X.shape).to(device)
    gen_img.weight.data.copy_(content_X.data)
    optimizer = torch.optim.Adam(gen_img.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_decay_epoch, 0.8)

    os.makedirs("results", exist_ok=True)
    losses = []
    fig, ax = plt.subplots()
    plt.ion()

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        X = gen_img()
        contents_Y_hat, styles_Y_hat = extract_features(X, net, content_layers, style_layers)
        loss = compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram,
                            content_weight, style_weight, tv_weight)
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}, Loss {loss.item():.2f}")
            ax.clear()
            ax.plot(losses, label="Loss")
            ax.set_title("Strong Style Transfer Loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()
            plt.pause(0.001)

    plt.ioff()
    result = postprocess(gen_img())
    content_name = os.path.basename(content_path).split('.')[0]
    style_name = os.path.basename(style_path).split('.')[0]
    result_img_path = f"results/STRONG_result_{content_name}_{style_name}.png"
    result_graph_path = f"results/STRONG_result_{content_name}_{style_name}_loss.png"
    result.save(result_img_path)
    plt.savefig(result_graph_path)
    print(f"Saved: {result_img_path}")
    print(f"Saved: {result_graph_path}")

# ─────────────── utils for folder scanning ───────────────
def list_images(dir_path, exts=None):
    if exts is None:
        exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        print(f"[INFO] Created empty directory: {dir_path}")
        return []
    files = [os.path.join(dir_path, f) for f in os.listdir(dir_path)
             if f.lower().endswith(exts) and os.path.isfile(os.path.join(dir_path, f))]
    files.sort()
    return files

# ─────────────── execute ───────────────
if __name__ == "__main__":
    use_strong_style = True  # Choose whether to highlight style here

    # New: scan folders instead of hardcoding "img/..."
    content_dir = "content_images"
    style_dir = "style_images"

    content_images = list_images(content_dir)
    style_images = list_images(style_dir)

    if not content_images:
        print(f"[WARN] No images found in '{content_dir}'. Put content images there.")
    if not style_images:
        print(f"[WARN] No images found in '{style_dir}'. Put style images there.")

    if content_images and style_images:
        for c_img in content_images:
            for s_img in style_images:
                if use_strong_style:
                    run_style_transfer_strong_style(c_img, s_img)
                else:
                    run_style_transfer(c_img, s_img)
    else:
        print("[STOP] Nothing to process. Exiting.")
