from focusui.visualization import draw_bbox

from PIL import Image, ImageDraw, ImageColor

# 1. 加载图片（注意：函数内部使用了 alpha_composite，所以原始图片建议转为 RGBA）
img = Image.open("./datasets/Example-Data/images/1c6422e3-8eea-44db-9d70-67e74920ae02.png").convert("RGBA")

# 2. 定义目标框 [x1, y1, x2, y2]
# x1, y1 是左上角坐标；x2, y2 是右下角坐标
my_bbox = [
                    0.098,
                    0.762,
                    0.269,
                    0.829
                ]

# 3. 调用函数
# color 可以传颜色名称，如 "blue", "green", "red" 等
result_img = draw_bbox(img, my_bbox, color="blue")

# 4. 保存或显示
result_img.show()
result_img.save("output.jpg")


