# PyTorch Style Transfer

This project implements **Neural Style Transfer** using **PyTorch** and a pre-trained **VGG19** model.  
It combines a **content image** with a **style image** to produce a new image where the content is preserved but the style is applied.

---

## 🎨 About Style Images
For the best results, it is highly recommended to use **style images with strong and distinctive artistic features**.  
Examples include:
- Van Gogh’s *The Starry Night* (Public Domain)
- Watercolor-style paintings
- Highly textured abstract art

Using style images with clear, prominent features allows the neural network to apply the artistic style more effectively.

---

## 📌 Example Result
*(Example images should be Public Domain or created by yourself)*

| Content Image | Style Image | Result |
|---------------|-------------|--------|
| ![content](examples/content.jpg) | ![style](examples/style.jpg) | ![result](examples/result.png) |

---

## 📂 Project Structure
pytorch-style-transfer/
├── style_transfer.py # Main code
├── init_folders.py # Script to auto-create required folders
├── content_images/ # Content images (empty by default)
├── style_images/ # Style images (empty by default)
├── results/ # Output results
├── .gitignore
└── README.md
---

## ⚙️ Installation
pip install torch torchvision pillow matplotlib

▶ How to Run
1. Place your images:
- Put content images in the content_images/ folder.
- Put style images in the style_images/ folder.

2. Run the script in the terminal:
- python style_transfer.py

3. The program will process all combinations of content and style images, and save:
- The result image with the style applied.
- The loss curve plot showing training progress.

---

## Example Terminal Output
Epoch 50, Loss 1.63
Epoch 100, Loss 1.39
Epoch 150, Loss 1.30
...
Epoch 1000, Loss 1.13
Saved: results/STRONG_result_content_03_style_03.png
Saved: results/STRONG_result_content_03_style_03_loss.png

---
⚠️ Notes
Image copyright: The code is distributed without any copyrighted images.
Use only Public Domain or CC0 images, or your own creations.

Well-known artworks by artists who died more than 70 years ago (e.g., Van Gogh) are Public Domain,
but photographs from museums or galleries may have their own restrictions.

If you use The Starry Night, download it from a Public Domain source such as Wikimedia Commons.