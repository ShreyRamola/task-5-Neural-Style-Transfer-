import torch
import torchvision.transforms as transforms
from torchvision.models import vgg19
from PIL import Image

device = torch.device("cpu")

vgg = vgg19(pretrained=True).features.to(device).eval()
for param in vgg.parameters():
    param.requires_grad = False

def load_image(path):
    image = Image.open(path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device)

def im_convert(tensor):
    image = tensor.cpu().clone().detach().squeeze(0)
    image = image * torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    image += torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    return transforms.ToPILImage()(image.clamp(0,1))

def get_features(image, model):
    layers = {'0': 'conv1_1', '5': 'conv2_1', '10': 'conv3_1',
              '19': 'conv4_1', '21': 'conv4_2', '28': 'conv5_1'}
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    tensor = tensor.view(c, h * w)
    return torch.mm(tensor, tensor.t())

# Load images
content = load_image("content.jpg")
style = load_image("style.jpg")

# Get features
content_features = get_features(content, vgg)
style_features = get_features(style, vgg)
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

# Optimize target image
target = content.clone().requires_grad_(True).to(device)
optimizer = torch.optim.Adam([target], lr=0.003)

style_weight = 1e4
content_weight = 1e0

print("Starting style transfer...")
for i in range(100):
    target_features = get_features(target, vgg)

    content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)

    style_loss = 0
    for layer in style_grams:
        target_gram = gram_matrix(target_features[layer])
        style_gram = style_grams[layer]
        _, d, h, w = target_features[layer].shape
        layer_style_loss = torch.mean((target_gram - style_gram)**2)
        style_loss += layer_style_loss / (d * h * w)  # scaled to avoid explosion


    total_loss = content_weight * content_loss + style_weight * style_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if i % 10 == 0:
        print(f"Step {i}, Total Loss: {total_loss.item():.2f}")

im_convert(target).save("stylized_output.jpg")
print("Saved as stylized_output.jpg")
