import random
from PIL import Image, ImageDraw, ImageFont


def randRGB():
    return (random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255))


def randPoint(size):
    (width, height) = size
    return (random.randint(0, width), random.randint(0, height))


class ImageChar():
    def __init__(self, size=(100, 40)):
        self.size = size
        # self.fontPath = fontPath
        self.bgColor = randRGB()
        self.image = Image.new('RGB', size, self.bgColor)

    def rotate(self):
        self.image.rotate(random.randint(0, 30), expand=0)

    def randLine(self, num):
        draw = ImageDraw.Draw(self.image)
        for i in range(0, num):
            draw.line([self.randPoint(), self.randPoint()], self.randRGB())
        del draw

    def save(self, path):
        self.image.save(path)


if __name__ == '__main__':
    ic = ImageChar(fontColor=(100, 211, 90))
    ic.save("5.jpeg")
