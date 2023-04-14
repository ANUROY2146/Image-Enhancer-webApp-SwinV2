import os.path as osp
import os
import sys
import glob
import cv2 as cv
import numpy as np
import torch
import SwinV2_arch as arch
from flask import Flask, render_template
from math import log10, sqrt
from skimage.metrics import structural_similarity

app = Flask(__name__)

model_path = 'models/SwinV2_4X.pth'
device = torch.device('cuda')

test_img_folder = 'LR/*'

model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)

#print('Model path models/SwinV2.pth \nProcessing...')

@app.route("/")
@app.route("/home")
def home():
    return render_template("Index.html", p1="static/results/1.png", p2="static/results/2.png", p3="static/results/3.png", p4="static/results/4.png", p5="static/results/5.png", cloud="static/results/clouds.png", fog="static/results/fog.png", title="static/results/TITLE_Background.jpg")
@app.route("/result", methods = ['POST', 'GET'])
def result():
    idx = 0
    for path in glob.glob(test_img_folder):
        idx += 1
        base = osp.splitext(osp.basename(path))[0]
        print(idx, base)
        # read images
        img = cv.imread(path, cv.IMREAD_COLOR)
        #cv.resize(img,(120,120))
        img = img * 1.0 / 255
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_LR = img.unsqueeze(0)
        img_LR = img_LR.to(device)

        with torch.no_grad():
            output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round()
        cv.imwrite('static/final.png'.format(base), output)
        os.replace(path, 'static/input.png')


    original = cv.imread("static/final.png")
    compress = cv.imread("static/input.png", 1)
    h, w, _ = original.shape
    compressed = cv.resize(compress, (h, w))

    # PSNR
    mse = np.mean((original - compressed) ** 2)
    if (mse == 0):
        psnr = 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    print(f"PSNR value is {psnr} dB")

    # SSIM
    before_gray = cv.cvtColor(original, cv.COLOR_BGR2GRAY)
    after_gray = cv.cvtColor(compressed, cv.COLOR_BGR2GRAY)
    (score, diff) = structural_similarity(before_gray, after_gray, full=True)
    print(f"SSIM value is {score} dB")


    ddepth = cv.CV_16S
    kernel_size = 3

    src = cv.imread('static/input.png', cv.IMREAD_COLOR) # Load an image
        # Check if image is loaded fine
    if src is None:
      print ('Error opening image')
      print ('Program Arguments: [image_name -- default image.png]')
      sys.exit()
    src = cv.GaussianBlur(src, (3, 3), 0)
    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    cv.imwrite('static/grey.png', src_gray)
    dst = cv.Laplacian(src_gray, ddepth, ksize=kernel_size)
    abs_dst = cv.convertScaleAbs(dst)

    cv.imwrite('static/hf1.png', abs_dst)
    cv.waitKey(0)
    cv.destroyAllWindows()


    ddepth = cv.CV_16S
    kernel_size = 3

    src = cv.imread('static/final.png', cv.IMREAD_COLOR) # Load an image
        # Check if image is loaded fine
    if src is None:
      print ('Error opening image')
      print ('Program Arguments: [image_name -- default image.png]')
      sys.exit()
    src = cv.GaussianBlur(src, (3, 3), 0)
    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    dst = cv.Laplacian(src_gray, ddepth, ksize=kernel_size)
    abs_dst = cv.convertScaleAbs(dst)

    cv.imwrite('static/hf2.png', abs_dst)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return render_template("Results.html", pic1="static/input.png", pic2="static/final.png", pic3="static/grey.png", pic4="static/hf1.png", pic5="static/hf2.png")

if __name__ == '__main__':
    app.run(debug = True, port = 5001)


