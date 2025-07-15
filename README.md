# 🖌️ Neural Style Transfer in PyTorch (CPU-Friendly)

This project applies the **artistic style of one image** (like a painting) to the **content of another image** (like a photo) using 🎨 Neural Style Transfer implemented in PyTorch.

It is designed to run **entirely on CPU** 🖥️ and uses a pre-trained **VGG19** model.

---

## 🔍 How It Works

- 🧠 Uses a pre-trained VGG19 CNN model to extract features from both **content** and **style** images.
- 📊 Computes **content loss** and **style loss** using Gram matrices.
- 🖼️ Optimizes a **target image** to minimize the combined loss.
- 🎯 Produces a final image that blends **content** and **style**.

---

## 📁 Files in This Repository

| 📄 File Name           | 📝 Description                          |
|------------------------|-----------------------------------------|
| `main.py`              | Python script that runs the style transfer |
| `content.jpg`          | Input content image (e.g. house)        |
| `style.jpg`            | Input style image (e.g. Van Gogh)       |
| `stylized_output.jpg`  | Output image after style transfer       |
| `README.txt`           | This file                               |

---

## 🖼️ Example Output

After the script finishes, it saves the output image as:

```
stylized_output.jpg
```

✅ The image contains the **content of `content.jpg`** painted in the **style of `style.jpg`**.

---

## 📚 Model Details

- 🧠 Uses `torchvision.models.vgg19(pretrained=True)`
- 🧩 Content features from: `conv4_2`
- 🎨 Style features from: `conv1_1`, `conv2_1`, ..., `conv5_1`
- 🧮 Optimized using Adam

---

## 🚀 CPU-Optimized

This implementation is lightweight and works well on machines without a GPU:

- 📏 All images are resized to `256x256`
- 🧠 Computation stays efficient with scaled style loss
- 💻 Runs completely on CPU

---

## 👤 Author

This project is a part of my deep learning experiments using PyTorch.  
> 🔬 Built for learning and demonstration purposes.

Feel free to fork, use, or improve it! 🚀
