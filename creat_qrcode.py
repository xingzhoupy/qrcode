import random
from PIL import Image
import os
import qrcode
import cv2
from scipy.misc import imread,imresize

# img=qrcode.make("some date here")
# img.save('H:/1.png')

def open_dir():
    imgs = []
    for root, dir, filename in os.walk(r'C:\Users\User\Desktop\bg', topdown=False):
        for name in filename:
            print(os.path.join(root, name))
            imgs.append(open_bimg(os.path.join(root, name)))
    return imgs


def open_bimg(path):
    return Image.open(path)


imgs = []


def get_bimages():
    global imgs
    imgs = open_dir()


def add_img(img, x, y):
    i = random.choice([i for i in range(len(imgs))])
    # for i in range(len(imgs)):
    base_img = imgs[i].resize((399, 260))

    # base_img = Image.open(r'C:\Users\User\Downloads\7272e452300cb37d635594be97356f32.jpg')
    # 可以查看图片的size和mode，常见mode有RGB和RGBA，RGBA比RGB多了Alpha透明度
    # print base_img.size, base_img.mode
    box = (210, 100, 330, 200)
    # if i == 15:
    #     box = (300, 200, 800, 800)  # 底图上需要P掉的区域
    # elif i in [10, 4, 14]:
    #     box = (280, 35, 350, 350)
    # elif i == 10:
    #     box = (250, 32, 340, 340)
    # elif i == 7:
    #     box = (700, 100, 1000, 1000)
    # elif i == 0:
    #     box = (210, 32, 340, 340)
    # elif i == 1:
    #     box = (240, 502, 700, 700)
    # elif i == 3:
    #     box = (250, 255, 400, 400)
    # elif i == 13:
    #     box = (250, 32, 340, 340)
    # elif i == 8:
    #     box = (20, 22, 50, 50)
    # elif i ==11:
    #     box = (250, 70, 400, 440)
    # else:
    #     box = (250, 32, 340, 340)
    # 加载需要P上去的图片
    # tmp_img = Image.open(r'D:\Desktop\2.png')
    # 这里可以选择一块区域或者整张图片
    # region = tmp_img.crop((0,0,304,546)) #选择一块区域
    # 或者使用整张图片
    # region = tmp_img
    # 使用 paste(region, box) 方法将图片粘贴到另一种图片上去.
    # 注意，region的大小必须和box的大小完全匹配。但是两张图片的mode可以不同，合并的时候回自动转化。如果需要保留透明度，则使用RGMA mode
    # 提前将图片进行缩放，以适应box区域大小
    # region = img.rotate(180) #对图片进行旋转
    region = img.resize((box[2] - box[0], box[3] - box[1]))
    print(region.size)
    # region = img.resize((box[2], box[3]))
    base_img.paste(region, box)
    # base_img.show()  # 查看合成的图片
    # base_img.save(r'H:/demo/' + str(i) + '.png') #保存图片
    return base_img


def init_qr(num):
    for i in range(num):
        box_size = random.choice([i for i in range(10, 20)])
        # rotate = random.choice([i for i in range(0, 180)])
        b1 = random.choice([i for i in range(1, 10)])
        b2= random.choice([i for i in range(1, 5)])
        # b3 = random.choice([i for i in range(1, 256)])
        # print(version, box_size, border)
        qr = qrcode.QRCode(
            version=b2,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=box_size,
            border=b1)
        qr.make(fit=True)
        img = qr.make_image()
        # region = img.transpose(Image.ROTATE_180)  # 翻转
        # img = img.rotate(rotate)  # 旋转
        # print(img)
        # im = Image.open('jb51.net.png')
        x, y = img.size
        # print(x, y)
        # img.resize((600, 600))
        # img = add_img(img, x, y)
        img = img.rotate(random.choice([i for i in range(0,30)]))
        img = img.resize((100, 100)).convert('L')
        # print(img.size)
        img.save(r'C:/Users/User/Desktop/demo/' + str(i) + '.jpg')
        # p = Image.new('RGBA', img.size, (255, 255, 255))
        # p.paste(img, (0, 0, x, y), img)
        # p.save(r'H:/demo/' + str(i) + '.png')


def init_bg(num):
    n = [i for i in range(len(imgs))]
    for i in range(num):
        a = random.choice(n)
        img = imgs[a].resize((399, 260))
        # img = img.rotate(random.choice([i for i in range(0, 180)]))
        # img = img.resize((399, 260))
        print(img.size)
        img.save(r'C:/tmp/Data/false/' + str(i) + '.png')
        # p = Image.new('RGBA', img.size, (255, 255, 255))
        # p.paste(img, (0, 0, x, y), img)
        # p.save(r'H:/demo/' + str(i) + '.png')


def open_image():
    count = 2990
    for root, dir, filename in os.walk(r'C:\Users\User\Desktop\demo', topdown=False):
        for name in filename:
            os.rename(os.path.join(root, name), "C:/Users/User/Desktop/true/" + str(count) + '.jpg')
            # print(os.path.join(root, name))
            print(count)
            count = 1 + count

def test():
    count = 2990
    for root, dir, filename in os.walk(r'C:\Users\User\Desktop\demo', topdown=False):
        for name in filename:
            img = open_bimg(os.path.join(root,name))
            img = img.convert('L')
            # img.show()
            img.save(r'C:/Users/User/Desktop/true/'+str(count)+".jpg")
            count = count+1




if __name__ == '__main__':
    # open_dir()
    # get_bimages()
    init_qr(3000)
    # init_bg(5000)
    # open_image()
    # test()