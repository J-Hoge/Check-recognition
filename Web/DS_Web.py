import io
import logging
import webbrowser

import cv2
import numpy as np
import paddleocr
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, request, render_template, send_file

logging.getLogger('werkzeug').disabled = True
# 禁用Flask开发服务器输出的日志信息，包括一些警告信息。这可以使Flask应用程序启动时不会在控制台输出过多的日志信息。

webbrowser.open('http://localhost:5000')  # 在默认浏览器中打开应用程序的首页。这可以帮助你快速测试你的应用程序，并且可以方便地在浏览器中查看应用程序的UI界面。

app = Flask(__name__)


def text_detection_SLOW(img):
    # 初始化PaddleOCR
    ocr = paddleocr.PaddleOCR(use_gpu=True, det_algorithm="DB", det_model_dir="OCR_Models/Final_Det",
                              det_max_side_len=7680, det_db_score_mode="slow", rec_model_dir="OCR_Models/Final_Rec",
                              max_text_length=100, use_space_char=True, use_angle_cls=True,
                              cls_model_dir="OCR_Models/Final_Angle_Cls")

    # 创建传入文件的拷贝
    image = img

    result = ocr.ocr(image, cls=True)
    with open('./out', 'w', encoding='utf-8') as f:
        for idx in range(len(result)):
            res = result[idx]
            for line in res:
                f.write(line[1][0] + '\n')

    # 获取结果
    result = result[0]

    # 创建Pillow可以用的图片
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # 设置字体
    font_path = 'Fonts/msyhl.ttc'
    # 设置字体大小
    msyhl_font = ImageFont.truetype(font_path, size=30)

    # 创建一个半透明图层
    draw = ImageDraw.Draw(image_pil)
    overlay = Image.new('RGBA', image_pil.size, (255, 255, 255, 0))
    draw_overlay = ImageDraw.Draw(overlay)

    # 在图片上画出识别出的内容
    for line in result:
        box = line[0]
        startX = int(box[0][0])
        startY = int(box[0][1])
        endX = int(box[2][0])
        endY = int(box[2][1])

        ouline_color = (0, 0, 256)
        outline_width = 3
        fill_color = (50, 50, 50, 128)  # 设置半透明填充颜色，alpha值128
        rectangle_area = [(startX, startY), (endX, endY)]
        # 在半透明图层上绘制矩形
        draw_overlay.rectangle(rectangle_area, outline=ouline_color, width=outline_width, fill=fill_color)

    # 将半透明图层与原始图像合并
    image_pil = Image.alpha_composite(image_pil.convert('RGBA'), overlay)

    # 将结果图像转换回 RGB 模式
    image_pil = image_pil.convert('RGB')

    # 在图像上绘制文本
    draw = ImageDraw.Draw(image_pil)
    for line in result:
        box = line[0]
        startX = int(box[0][0])
        startY = int(box[0][1])
        startXY = (startX, startY)

        txts = line[1][0]

        draw.text(startXY, str(txts), font=msyhl_font, fill=(0, 255, 0))

    image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    return image


def upscale_RealSR(image):
    # 对图像进行超分辨率 - 检查图像尺寸，如果小于 3000，则进行超分辨率处理
    width, height = image.size
    max_side = max(width, height)
    if max_side < 3000:
        # 进行超分辨率处理
        from ppgan.apps import RealSRPredictor
        sr = RealSRPredictor()
        image = sr.run(image)
        # 将PIL图像转换为numpy.ndarray
        image_np = np.array(image[0])
    else:
        # 直接处理 - 将PIL图像转换为ndarray
        image_np = np.array(image)

    return image_np


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # 获取上传的文件
        file = request.files['file']

        # 读取图像文件
        image = Image.open(file).convert('RGB')

        # 对分辨率过低的图像进行超分辨率
        image_np = upscale_RealSR(image)

        # 将文件送入OCR网络
        im2 = text_detection_SLOW(image_np)

        # 将处理后的numpy.ndarray图像转换回PIL图像
        im2_pil = Image.fromarray(im2)

        
         # 将图像文件内容存储到内存缓冲区中
        buffer = io.BytesIO()
        im2_pil.save(buffer, format='JPEG')
        encoded_string = base64.b64encode(buffer.getvalue()).decode('utf-8')

        with open('out.txt', 'r', encoding='utf-8') as f:
            content = f.read()
        # 返回图像文件内容
        return render_template('result.html', image_data=encoded_string, result=content)
    else:
        return render_template('index.html')



if __name__ == '__main__':
    app.run()
