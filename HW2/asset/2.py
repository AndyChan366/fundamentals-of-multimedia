# coding=utf-8
from PIL import Image
import imageio
import math
import numpy as np
MiniSize = 8
ZigzagMat = [0]
# 亮度量化矩阵
Matoflightquan = [
    16,11,10,16,24,40,51,61,
    12,12,14,19,26,58,60,55,
    14,13,16,24,40,57,69,56,
    14,17,22,29,51,87,80,62,
    18,22,37,56,68,109,103,77,
    24,35,55,64,81,104,113,92,
    49,64,78,87,103,121,120,101,
    72,92,95,98,112,100,103,99]
# 色度量化矩阵
Matofcolorquan = [
    17,18,24,47,99,99,99,99,
    18,21,26,66,99,99,99,99,
    24,26,56,99,99,99,99,99,
    47,66,99,99,99,99,99,99,
    99,99,99,99,99,99,99,99,
    99,99,99,99,99,99,99,99,
    99,99,99,99,99,99,99,99,
    99,99,99,99,99,99,99,99]
# DC系数亮度Huffman表
HuffmanofDClight = {
    0:  '000',
    1:  '010',
    2:  '011',
    3:  '100',
    4:  '101',
    5:  '110',
    6:  '1110',
    7:  '11110',
    8:  '111110',
    9:  '1111110',
    10: '11111110',
    11: '111111110'
}
# DC系数色度Huffman表
HuffmanofDCcolor = {
    0:  '00',
    1:  '01',
    2:  '10',
    3:  '110',
    4:  '1110',
    5:  '11110',
    6:  '111110',
    7:  '1111110',
    8:  '11111110',
    9:  '111111110',
    10: '1111111110',
    11: '11111111110'
}
# AC系数亮度Huffman表
HuffmanofAClight = {
    (0,0):'1010',
    (0,1):'00',
    (0,2):'01',
    (0,3):'100',
    (0,4):'1011',
    (0,5):'11010',
    (0,6):'1111000',
    (0,7):'11111000',
    (0,8):'1111110110',
    (0,9):'1111111110000010',
    (0,10):'1111111110000011',
    (1,1):'1100',
    (1,2):'11011',
    (1,3):'1111001',
    (1,4):'111110110',
    (1,5):'11111110110',
    (1,6):'1111111110000100',
    (1,7):'1111111110000101',
    (1,8):'1111111110000110',
    (1,9):'1111111110000111',
    (1,10):'1111111110001000',
    (2,1):'11100',
    (2,2):'11111001',
    (2,3):'1111110111',
    (2,4):'111111110100',
    (2,5):'1111111110001001',
    (2,6):'1111111110001010',
    (2,7):'1111111110001011',
    (2,8):'1111111110001100',
    (2,9):'1111111110001101',
    (2,10):'1111111110001110',
    (3,1):'111010',
    (3,2):'111110111',
    (3,3):'111111110101',
    (3,4):'1111111110001111',
    (3,5):'1111111110010000',
    (3,6):'1111111110010001',
    (3,7):'1111111110010010',
    (3,8):'1111111110010011',
    (3,9):'1111111110010100',
    (3,10):'1111111110010101',
    (4,1):'111011',
    (4,2):'1111111000',
    (4,3):'1111111110010110',
    (4,4):'1111111110010111',
    (4,5):'1111111110011000',
    (4,6):'1111111110011001',
    (4,7):'1111111110011010',
    (4,8):'1111111110011011',
    (4,9):'1111111110011100',
    (4,10):'1111111110011101',
    (5,1):'1111010',
    (5,2):'11111110111',
    (5,3):'1111111110011110',
    (5,4):'1111111110011111',
    (5,5):'1111111110100000',
    (5,6):'1111111110100001',
    (5,7):'1111111110100010',
    (5,8):'1111111110100011',
    (5,9):'1111111110100100',
    (5,10):'1111111110100101',
    (6,1):'1111011',
    (6,2):'111111110110',
    (6,3):'1111111110100110',
    (6,4):'1111111110100111',
    (6,5):'1111111110101000',
    (6,6):'1111111110101001',
    (6,7):'1111111110101010',
    (6,8):'1111111110101011',
    (6,9):'1111111110101100',
    (6,10):'1111111110101101',
    (7,1):'11111010',
    (7,2):'111111110111',
    (7,3):'1111111110101110',
    (7,4):'1111111110101111',
    (7,5):'1111111110110000',
    (7,6):'1111111110110001',
    (7,7):'1111111110110010',
    (7,8):'1111111110110011',
    (7,9):'1111111110110100',
    (7,10):'1111111110110101',
    (8,1):'111111000',
    (8,2):'111111111000000',
    (8,3):'1111111110110110',
    (8,4):'1111111110110111',
    (8,5):'1111111110111000',
    (8,6):'1111111110111001',
    (8,7):'1111111110111010',
    (8,8):'1111111110111011',
    (8,9):'1111111110111100',
    (8,10):'1111111110111101',
    (9,1):'111111001',
    (9,2):'1111111110111110',
    (9,3):'1111111110111111',
    (9,4):'1111111111000000',
    (9,5):'1111111111000001',
    (9,6):'1111111111000010',
    (9,7):'1111111111000011',
    (9,8):'1111111111000100',
    (9,9):'1111111111000101',
    (9,10):'1111111111000110',
    (10,1):'111111010',
    (10,2):'1111111111000111',
    (10,3):'1111111111001000',
    (10,4):'1111111111001001',
    (10,5):'1111111111001010',
    (10,6):'1111111111001011',
    (10,7):'1111111111001100',
    (10,8):'1111111111001101',
    (10,9):'1111111111001110',
    (10,10):'1111111111001111',
    (11,1):'1111111001',
    (11,2):'1111111111010000',
    (11,3):'1111111111010001',
    (11,4):'1111111111010010',
    (11,5):'1111111111010011',
    (11,6):'1111111111010100',
    (11,7):'1111111111010101',
    (11,8):'1111111111010110',
    (11,9):'1111111111010111',
    (11,10):'1111111111011000',
    (12,1):'1111111010',
    (12,2):'1111111111011001',
    (12,3):'1111111111011010',
    (12,4):'1111111111011011',
    (12,5):'1111111111011100',
    (12,6):'1111111111011101',
    (12,7):'1111111111011110',
    (12,8):'1111111111011111',
    (12,9):'1111111111100000',
    (12,10):'1111111111100001',
    (13,1):'11111111000',
    (13,2):'1111111111100010',
    (13,3):'1111111111100011',
    (13,4):'1111111111100100',
    (13,5):'1111111111100101',
    (13,6):'1111111111100110',
    (13,7):'1111111111100111',
    (13,8):'1111111111101000',
    (13,9):'1111111111101001',
    (13,10):'1111111111101010',
    (14,1):'1111111111101011',
    (14,2):'1111111111101100',
    (14,3):'1111111111101101',
    (14,4):'1111111111101110',
    (14,5):'1111111111101111',
    (14,6):'1111111111110000',
    (14,7):'1111111111110001',
    (14,8):'1111111111110010',
    (14,9):'1111111111110011',
    (14,10):'1111111111110100',
    (15,0):'11111111001',
    (15,1):'1111111111110101',
    (15,2):'1111111111110110',
    (15,3):'1111111111110111',
    (15,4):'1111111111111000',
    (15,5):'1111111111111001',
    (15,6):'1111111111111010',
    (15,7):'1111111111111011',
    (15,8):'1111111111111100',
    (15,9):'1111111111111101',
    (15,10):'1111111111111110'
}
# AC系数色度Huffman表
HuffmanofACcolor = {
    (0,0):'00',
    (0,1):'01',
    (0,2):'100',
    (0,3):'1010',
    (0,4):'11000',
    (0,5):'11001',
    (0,6):'111000',
    (0,7):'1111000',
    (0,8):'111110100',
    (0,9):'1111110110',
    (0,10):'111111110100',
    (1,1):'1011',
    (1,2):'111001',
    (1,3):'11110110',
    (1,4):'111110101',
    (1,5):'11111110110',
    (1,6):'111111110101',
    (1,7):'1111111110001000',
    (1,8):'1111111110001001',
    (1,9):'1111111110001010',
    (1,10):'1111111110001011',
    (2,1):'11010',
    (2,2):'11110111',
    (2,3):'1111110111',
    (2,4):'111111110110',
    (2,5):'111111111000010',
    (2,6):'1111111110001100',
    (2,7):'1111111110001101',
    (2,8):'1111111110001110',
    (2,9):'1111111110001111',
    (2,10):'1111111110010000',
    (3,1):'11011',
    (3,2):'11111000',
    (3,3):'1111111000',
    (3,4):'111111110111',
    (3,5):'1111111110010001',
    (3,6):'1111111110010010',
    (3,7):'1111111110010011',
    (3,8):'1111111110010100',
    (3,9):'1111111110010101',
    (3,10):'1111111110010110',
    (4,1):'111010',
    (4,2):'111110110',
    (4,3):'1111111110010111',
    (4,4):'1111111110011000',
    (4,5):'1111111110011001',
    (4,6):'1111111110011010',
    (4,7):'1111111110011011',
    (4,8):'1111111110011100',
    (4,9):'1111111110011101',
    (4,10):'1111111110011110',
    (5,1):'111011',
    (5,2):'1111111001',
    (5,3):'1111111110011111',
    (5,4):'1111111110100000',
    (5,5):'1111111110100001',
    (5,6):'1111111110100010',
    (5,7):'1111111110100011',
    (5,8):'1111111110100100',
    (5,9):'1111111110100101',
    (5,10):'1111111110100110',
    (6,1):'1111001',
    (6,2):'11111110111',
    (6,3):'1111111110100111',
    (6,4):'1111111110101000',
    (6,5):'1111111110101001',
    (6,6):'1111111110101010',
    (6,7):'1111111110101011',
    (6,8):'1111111110101100',
    (6,9):'1111111110101101',
    (6,10):'1111111110101110',
    (7,1):'1111010',
    (7,2):'11111111000',
    (7,3):'1111111110101111',
    (7,4):'1111111110110000',
    (7,5):'1111111110110001',
    (7,6):'1111111110110010',
    (7,7):'1111111110110011',
    (7,8):'1111111110110100',
    (7,9):'1111111110110101',
    (7,10):'1111111110110110',
    (8,1):'11111001',
    (8,2):'1111111110110111',
    (8,3):'1111111110111000',
    (8,4):'1111111110111001',
    (8,5):'1111111110111010',
    (8,6):'1111111110111011',
    (8,7):'1111111110111100',
    (8,8):'1111111110111101',
    (8,9):'1111111110111110',
    (8,10):'1111111110111111',
    (9,1):'111110111',
    (9,2):'1111111111000000',
    (9,3):'1111111111000001',
    (9,4):'1111111111000010',
    (9,5):'1111111111000011',
    (9,6):'1111111111000100',
    (9,7):'1111111111000101',
    (9,8):'1111111111000110',
    (9,9):'1111111111000111',
    (9,10):'1111111111001000',
    (10,1):'111111000',
    (10,2):'1111111111001001',
    (10,3):'1111111111001010',
    (10,4):'1111111111001011',
    (10,5):'1111111111001100',
    (10,6):'1111111111001101',
    (10,7):'1111111111001110',
    (10,8):'1111111111001111',
    (10,9):'1111111111010000',
    (10,10):'1111111111010001',
    (11,1):'111111001',
    (11,2):'1111111111010010',
    (11,3):'1111111111010011',
    (11,4):'1111111111010100',
    (11,5):'1111111111010101',
    (11,6):'1111111111010110',
    (11,7):'1111111111010111',
    (11,8):'1111111111011000',
    (11,9):'1111111111011001',
    (11,10):'1111111111011010',
    (12,1):'111111010',
    (12,2):'1111111111011011',
    (12,3):'1111111111011100',
    (12,4):'1111111111011101',
    (12,5):'1111111111011110',
    (12,6):'1111111111011111',
    (12,7):'1111111111100000',
    (12,8):'1111111111100001',
    (12,9):'1111111111100010',
    (12,10):'1111111111100011',
    (13,1):'11111111001',
    (13,2):'1111111111100100',
    (13,3):'1111111111100101',
    (13,4):'1111111111100110',
    (13,5):'1111111111100111',
    (13,6):'1111111111101000',
    (13,7):'1111111111101001',
    (13,8):'1111111111101010',
    (13,9):'1111111111101011',
    (13,10):'1111111111101100',
    (14,1):'11111111100000',
    (14,2):'1111111111101101',
    (14,3):'1111111111101110',
    (14,4):'1111111111101111',
    (14,5):'1111111111110000',
    (14,6):'1111111111110001',
    (14,7):'1111111111110010',
    (14,8):'1111111111110011',
    (14,9):'1111111111110100',
    (14,10):'1111111111110101',
    (15,0):'1111111010',
    (15,1):'111111111000011',
    (15,2):'1111111111110110',
    (15,3):'1111111111110111',
    (15,4):'1111111111111000',
    (15,5):'1111111111111001',
    (15,6):'1111111111111010',
    (15,7):'1111111111111011',
    (15,8):'1111111111111100',
    (15,9):'1111111111111101',
    (15,10):'1111111111111110'
}

# 计算两张图片的差别，用均方误差衡量
def compute_mse(X, Y):
    X = np.float32(X)
    Y = np.float32(Y)
    mse = np.mean((X - Y) ** 2, dtype=np.float64)
    return mse

# 判断是否越界
def Boundary(x, y):
    if x < 0 or y < 0 or x >= MiniSize or y >= MiniSize:
        return False
    else:
        return True

# 十进制转二进制
def dec2bin(Number):
    result = bin(Number)
    res = result.replace('0b','')
    return res

# 二进制转十进制
def bin2dec(Number):
    return float(int(Number,2))

# 供主函数调用，顺序压缩两个所给图片
def load(f):
    if f == 1:
        img = Image.open("动物照片.jpg")
        img.convert('RGB').save('animal.bmp')
        image = Image.open("animal.bmp")
    elif f == 2:
        img = Image.open("动物卡通图片.jpg")
        img.convert('RGB').save('cartoon.bmp')
        image = Image.open("cartoon.bmp")
    return image

# 格式化输出提示信息，见主函数
def seq(i):
    if i == 1: return "st"
    elif i == 2: return "nd"

# 填充，分块为8个一组，因此要保证size为8的倍数
def Fill(matrix, size, newsize):
    newsize0 = size[0]
    newsize1 = size[1]
    fillsize1 = size[1] % 8
    if fillsize1 != 0:
        newsize1 = size[1] + 8 - fillsize1
    fillsize0 = size[0] % 8
    if fillsize0 != 0:
        newsize0 = size[0] + 8 - fillsize0
    newsize[0] = newsize0
    newsize[1] = newsize1
    temp = []
    for i in range(0, newsize1):
        for j in range(0, newsize0):
            if i < size[1] and j < size[0]:
                temp.append(matrix[i * size[0] + j])
            else:
                temp.append(0)
    return temp

# 二次采样，采样方式为4:2:2。Y不动，对于U和V，两个一组相同。
def Subsample(color, size):
    temp = []
    if size[0] % 2 == 0:
        judge = size[0] // 2
    else:
        judge = (size[0] // 2) + 1
    for i in range(0, size[1]):
        for j in range(0, judge):
            begin = i * size[0] + j * 2
            if begin == size[0] - 1:
                # 在边界处
                temp.append(color[begin])
            else:
                temp.append(color[begin])
                temp.append(color[begin])
    return temp

# 分块，8个一组
def Block(matrix, sizeX, sizeY):
    temp = []
    for Y in range(0, sizeY // MiniSize):
        for X in range(0, sizeX // MiniSize):
            begin = Y * sizeX * 8 + X * 8
            immediate = []
            for i in range(0, MiniSize):
                for j in range(0, MiniSize):
                    immediate.append(matrix[begin + i * sizeX + j])
            temp.append(immediate)
    return temp

# DCT的系数
def coeff(u, MiniSize):
    return math.sqrt(1.0 / MiniSize) if u == 0 else math.sqrt(2.0 / MiniSize)

# DCT公式
def DCT(matrix, DCTMat, MiniSize):
    for x in range(0, MiniSize):
        for y in range(0, MiniSize):
            res = 0
            for i in range(0, MiniSize):
                for j in range(0, MiniSize):
                    res += math.cos((2 * i + 1) * x * math.pi / (2 * MiniSize)) * math.cos((2 * j + 1) * y * math.pi / (2 * MiniSize)) * matrix[i * MiniSize + j]
            DCTMat.append(int(round(coeff(x, MiniSize) * coeff(y, MiniSize) * res)))

# 亮度量化函数
def QuanofLight(DCTMat, QuanMat, MiniSize):
    for i in range(0, MiniSize * MiniSize):
        QuanMat.append(int(round(1.0 * DCTMat[i] / Matoflightquan[i])))

# 色度量化函数
def QuanofColor(DCTMat, QuanMat, MiniSize):
    for i in range(0, MiniSize * MiniSize):
        QuanMat.append(int(round(1.0 * DCTMat[i] / Matofcolorquan[i])))

# DCT&&量化函数接口
def DCTANDQuan(matrix, LorC):
    DCTMat = []
    QuanMat = []
    # 调用DCT函数生成DCT矩阵
    DCT(matrix, DCTMat, MiniSize)
    # 判断进行亮度还是色度量化
    if LorC == 0:
        QuanofLight(DCTMat, QuanMat, MiniSize)
    else:
        QuanofColor(DCTMat, QuanMat, MiniSize)
    return DCTMat

# 对每个块DCT+量化过程
def DCTFinals(BlockMat, length, LorC):
    temp = []
    flag = 0
    for i in range(0, length):
        flag += 1
        temp.append(DCTANDQuan(BlockMat[i], LorC))
    return temp

# Zigzag扫描，由于其移动过程有规律可循，因此操作索引即可
def Zigzag(matrix, ZigzagMat):
    temp = []
    i = 0
    j = 0
    # flag为0代表向右上方移动，1代表向左下方移动
    flag = 0
    temp.append(matrix[0])
    for k in range(0, MiniSize * MiniSize - 1):
        # 要考虑在边界时如何移动的情况
        if flag == 0:
            if Boundary(i + 1, j - 1):
                i = i + 1
                j = j - 1
            elif Boundary(i + 1, j):
                i = i + 1
                flag = 1
            elif Boundary(i, j + 1):
                j = j + 1
                flag = 1
        elif flag == 1:
            if Boundary(i - 1, j + 1):
                i = i - 1
                j = j + 1
            elif Boundary(i, j + 1):
                j = j + 1
                flag = 0
            elif Boundary(i + 1, j):
                i = i + 1
                flag = 0
        if len(ZigzagMat) != 64:
            ZigzagMat.append(j * MiniSize + i)
        temp.append(matrix[j * MiniSize + i])
    return temp

# DPCM&&游长编码
def EncodingofDCAC(DCACMat, ZigzagafterBlock, length):
    for i in range(0, length):
        temp = []
        # 用前后两块的差值求出DC系数
        if i == 0:
            temp.append((0, ZigzagafterBlock[i][0]))
        else:
            temp.append((0, ZigzagafterBlock[i][0] - ZigzagafterBlock[i - 1][0]))
        # 求出AC系数
        AC(temp, ZigzagafterBlock[i])
        DCACMat.append(temp)

# AC系数
def AC(immediate, initial):
    # 统计0的个数
    count = 0
    for i in range(1, MiniSize * MiniSize):
        if initial[i] != 0:
            immediate.append((count, initial[i]))
            count = 0
        elif i == MiniSize * MiniSize - 1:
            immediate.append((count, initial[i]))
            count = 0
        else:
            count += 1
        # 连续的0超过16个，则用（15，0）表示16连续的0
        if count > 15:
            immediate.append((15, 0))
            count = 0

# 输入为一个数字，返回它的VLI
def VLI(VLINum):
    input = dec2bin(VLINum)
    output = ''
    if VLINum < 0:
        for i in range(0, len(input)):
            if input[i] == '0':
                output = output + '1'
            elif input[i] == '1':
                output = output + '0'
    else:
        output = input
    return ((len(output), output))

# 哈夫曼编码
def HuffmanEncoding(CodingMat1, CodingMat2, LorC):
    for i in range(0, len(CodingMat1)):
        # DC部分的编码（矩阵的第一列为DC系数）
        (DCindex, VLIofDC) = VLI(CodingMat1[i][0][1])
        DCHUFFMAN = ''
        # 对于DC,判断是色度还是亮度量化
        if LorC == 0:
            DCHUFFMAN = HuffmanofDClight[DCindex]
        else:
            DCHUFFMAN = HuffmanofDCcolor[DCindex]
        finals = DCHUFFMAN + VLIofDC
        for j in range(1, len(CodingMat1[i])):
            # AC部分的编码（后面的为AC系数）
            (ACindex, VLIofAC) = VLI(CodingMat1[i][j][1])
            ACHUFFMAN = ''
            # 对于AC,判断是色度还是亮度量化
            if LorC == 0:
                ACHUFFMAN = HuffmanofAClight[(CodingMat1[i][j][0]), ACindex]
            else:
                ACHUFFMAN = HuffmanofACcolor[(CodingMat1[i][j][0]), ACindex]
            # finals存储每个8*8矩阵压缩后的编码
            finals += ACHUFFMAN + VLIofAC
        CodingMat2.append(finals)
# ——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# 解码过程
# 输入为str，第一个数字如果为0说明是负数，否则为正数，输出为一个数字
def IVLI(VLIStr):
    if VLIStr[0] == '0':
        Num_str = ''
        for i in range(0,len(VLIStr)):
            if VLIStr[i] == '1':
                Num_str += '0'
            elif VLIStr[i] == '0':
                Num_str += '1'
        return -bin2dec(Num_str)
    else:
        return bin2dec(VLIStr)

# 哈夫曼解码，由VLI转为数字
def IHuffmanEncoding(CodingMat2, CodingMat1, LorC):
    items = 0
    key = 0
    for i in range(0, len(CodingMat2)):
        j = 0
        # flag=0则代表DPCM，否则是游长编码
        flag = 0
        Num_str = ''
        immediate = []
        while j < len(CodingMat2[i]):
            Num_str += CodingMat2[i][j]
            # print "Num_str: %s" %(Num_str)
            j += 1
            # LorC判断是亮度还是色度
            if LorC == 0 :
                if flag == 0:
                    items = HuffmanofDClight.items()
                elif flag == 1:
                    items = HuffmanofAClight.items()
            elif LorC == 1:
                if flag == 0:
                    items = HuffmanofDCcolor.items()
                elif flag == 1:
                    items = HuffmanofACcolor.items()
            for (k,v) in items:
                # 二进制串匹配到了哈夫曼DC(AC)色度(亮度)编码表，则将对应的数（数对）存到key变量内（可参见最前面的编码表）
                if v == Num_str:
                    key = k
                    Num_str = ''
                    VLINum = 0
                    # 如果flag为0，那么说明要查哈夫曼DC编码表，此时key为一个数
                    if flag == 0:
                        VLIStr = ''
                        for k in range(0, key):
                            VLIStr += CodingMat2[i][j]
                            j += 1
                        # 将VLIStr变成整数
                        if VLIStr == '':
                            immediate.append((0, 0))
                        else:
                            VLINum = IVLI(VLIStr)
                            immediate.append((0, VLINum))
                    # 如果flag为1，那么说明要查哈夫曼AC编码表，此时key为数对（int，int）
                    elif flag == 1:
                        VLIStr = ''
                        for k in range(0, key[1]):
                            VLIStr += CodingMat2[i][j]
                            j+=1
                        if VLIStr == '':
                            immediate.append((key[0], 0))
                        else:
                            VLINum = IVLI(VLIStr)
                            immediate.append((key[0], VLINum))
                    flag = 1
        CodingMat1.append(immediate)

# 解码DPCM&&游长编码，为DPCM&&游长编码的逆过程，反推即可
def IEncodingofDCAC(DCACMat, ZigzagafterBlock):
    for i in range(0, len(DCACMat)):
        temp = []
        for j in range(0, len(DCACMat[i])):
            for k in range(0, DCACMat[i][j][0]):
                temp.append(0)
            if i > 0 and j == 0:
                temp.append(DCACMat[i][j][1] + ZigzagafterBlock[i - 1][0])
            else:
                temp.append(DCACMat[i][j][1])
        if len(temp) != 64:
            print (DCACMat[i])
            print ("Block number has error!")
            print (len(temp))
        ZigzagafterBlock.append(temp)

# Zigzag扫描的逆过程
def IZigzag(ZigzagafterBlock, ZigzagMat):
    temp = []
    if len(ZigzagafterBlock) != 64:
        print ("error!")
        print (len(ZigzagafterBlock))
    for j in range(0, len(ZigzagafterBlock)):
        for k in range(0, len(ZigzagMat)):
            if ZigzagMat[k] == j:
                temp.append(ZigzagafterBlock[k])
    if len(temp) != 64:
        print ("error!")
        print (len(temp))
    return temp

# DCT的逆变换，将DCT倒过来即可
def IDCTFinals(QuanMat, length, LorC):
    temp = []
    flag = 0
    for i in range(0, length):
        flag += 1
        temp.append(IDCTANDQuan(QuanMat[i], LorC))
    return temp

def IDCTANDQuan(QuanMat, LorC):
    DCTMat = []
    # 生成DCT矩阵
    if LorC == 0:
        IQuanofLight(DCTMat, QuanMat)
    else:
        IQuanofColor(DCTMat, QuanMat)
    # 逆量化,然后输出
    IDCTMat = IDCT(QuanMat)
    for i in range(0, MiniSize * MiniSize):
        IDCTMat[i] = int(round(IDCTMat[i]))
    return IDCTMat

# 亮度逆量化
def IQuanofLight(DCTMat, QuanMat):
    for i in range(0, MiniSize * MiniSize):
        DCTMat.append(QuanMat[i] * Matoflightquan[i])

# 色度逆量化
def IQuanofColor(DCTMat, QuanMat):
    for i in range(0, MiniSize * MiniSize):
        DCTMat.append(QuanMat[i] * Matofcolorquan[i])

# 逆DCT公式
def IDCTFormula(a, i):
    res = 0
    for j in range(0, MiniSize):
        res += math.cos((2 * i + 1) * j * math.pi / 16) * a[j] * coeff(j, MiniSize)
    return res

def IDCT(DCTMat):
    temp1 = list(range(64))
    temp2 = list(range(64))
    # 对每一列IDCT求出temp1
    for i in range(0, MiniSize):
        a = []
        for j in range(0, MiniSize):
            a.append(DCTMat[i + j * MiniSize])
        for k in range(0, MiniSize):
            temp1[k * MiniSize + i] = IDCTFormula(a, k)
    # 对每一行IDCT求出temp2
    for i in range(0, MiniSize):
        a = []
        for j in range(0, MiniSize):
            a.append(temp1[i * MiniSize + j])
        for k in range(0, MiniSize):
            temp2[i * MiniSize + k] = IDCTFormula(a, k)
    return temp2

# 合并块
def IBlock(BlockMat, size):
    temp = []
    for Y in range(0, size[1] // MiniSize):
        for row in range(0, MiniSize):
            for X in range(0, size[0] // MiniSize):
                for col in range(0, MiniSize):
                    if X + Y * (size[0] // MiniSize) >= len(BlockMat):
                        print ("?0")
                    elif col + row * MiniSize >= len(BlockMat[X + Y * (size[0] // MiniSize)]):
                        print ("?1 {} {} {} {} {} {}" .format(X, Y, row, col, col + row * MiniSize, len(BlockMat[X + Y * (size[0] // MiniSize)])))
                    temp.append(BlockMat[X + Y * (size[0] // MiniSize)][col + row * MiniSize])
    return temp

# 将图像缩小回原来大小
def IFill(matrix, size):
    newsize0 = size[0]
    newsize1 = size[1]
    fillsize1 = size[1] % 8
    if fillsize1 != 0:
        newsize1 = size[1] + 8 - fillsize1
    fillsize0 = size[0] % 8
    if fillsize0 != 0:
        newsize0 = size[0] + 8 - fillsize0
    temp= []
    for i in range(0, newsize1):
        for j in range(0, newsize0):
            if i < size[1] and j < size[0]:
                temp.append(matrix[i * newsize0 + j])
    return temp

if __name__=="__main__":
    count = 1
    while count < 3:
        # 读取文件
        image = load(count)
        print("Please wait about 5 minutes...The {}{} picture is compressing...".format(count, seq(count)))
        size = image.size
        # RGB->YUV
        Y = []
        U = []
        V = []
        # 根据RGB->YUV公式
        for i in range(0, size[0]):
            for j in range(0, size[1]):
                pixel = image.getpixel((i, j))
                Y.append(0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2])
                U.append(-0.169 * pixel[0] - 0.331 * pixel[1] + 0.5 * pixel[2] + 128)
                V.append(0.5 * pixel[0] - 0.419 * pixel[1] - 0.081 * pixel[2] + 128)
        # 采样，此处使用的比例是4:2:2
        U = Subsample(U, size)
        V = Subsample(V, size)
        # 填充
        newsize = [0, 0]
        Y = Fill(Y, size, newsize)
        U = Fill(U, size, newsize)
        V = Fill(V, size, newsize)
        # 分块，后续DCT要用
        YafterBlock = Block(Y, newsize[0], newsize[1])
        UafterBlock = Block(U, newsize[0], newsize[1])
        VafterBlock = Block(V, newsize[0], newsize[1])
        # 对每个块进行DCT和量化处理，第三个参数0代表亮度，1代表色度
        YafterBlock = DCTFinals(YafterBlock, len(YafterBlock), 0)
        UafterBlock = DCTFinals(UafterBlock, len(UafterBlock), 1)
        VafterBlock = DCTFinals(VafterBlock, len(VafterBlock), 1)
        # 先进行Zigzag扫描排序
        YBlockZigzag = []
        UBlockZigzag = []
        VBlockZigzag = []
        for i in range(0, len(YafterBlock)):
            YBlockZigzag.append(Zigzag(YafterBlock[i], ZigzagMat))
        for i in range(0, len(UafterBlock)):
            UBlockZigzag.append(Zigzag(UafterBlock[i], ZigzagMat))
        for i in range(0, len(VafterBlock)):
            VBlockZigzag.append(Zigzag(VafterBlock[i], ZigzagMat))
        # DC,AC系数编码
        DCACofY = []
        DCACofU = []
        DCACofV = []
        EncodingofDCAC(DCACofY, YBlockZigzag, len(YBlockZigzag))
        EncodingofDCAC(DCACofU, UBlockZigzag, len(UBlockZigzag))
        EncodingofDCAC(DCACofV, VBlockZigzag, len(VBlockZigzag))
        # 熵编码
        # 首先进行VLI变长整数编码，(x,y) = VLI
        # 需要三个表：VLI表、DC亮度Huffman表、AC亮度Huffman表
        # 用来存储最终的各个8*8矩阵的数据流
        EntCodingofY = []
        EntCodingofU = []
        EntCodingofV = []
        # 哈夫曼编码，第三个参数0代表亮度，1代表色度
        HuffmanEncoding(DCACofY, EntCodingofY, 0)
        HuffmanEncoding(DCACofU, EntCodingofU, 1)
        HuffmanEncoding(DCACofV, EntCodingofV, 1)
        # 解码过程，和编码过程相反
        # 从二进制串变成VLI数对
        IDCACofY = []
        IDCACofU = []
        IDCACofV = []
        IHuffmanEncoding(EntCodingofY, IDCACofY, 0)
        IHuffmanEncoding(EntCodingofU, IDCACofU, 1)
        IHuffmanEncoding(EntCodingofV, IDCACofV, 1)
        # 数对变成64位矩阵
        YBlockZigzag = []
        UBlockZigzag = []
        VBlockZigzag = []
        IEncodingofDCAC(DCACofY, YBlockZigzag)
        IEncodingofDCAC(DCACofU, UBlockZigzag)
        IEncodingofDCAC(DCACofV, VBlockZigzag)
        # 逆Zigzag
        YafterBlock = []
        UafterBlock = []
        VafterBlock = []
        for i in range(0, len(YBlockZigzag)):
            YafterBlock.append(IZigzag(YBlockZigzag[i], ZigzagMat))
        for i in range(0, len(UBlockZigzag)):
            UafterBlock.append(IZigzag(UBlockZigzag[i], ZigzagMat))
        for i in range(0, len(VBlockZigzag)):
            VafterBlock.append(IZigzag(VBlockZigzag[i], ZigzagMat))
        # 逆量化和IDCT
        YafterBlock = IDCTFinals(YafterBlock, len(YafterBlock), 0)
        UafterBlock = IDCTFinals(UafterBlock, len(UafterBlock), 1)
        VafterBlock = IDCTFinals(VafterBlock, len(VafterBlock), 1)
        # 块合并
        Y = IBlock(YafterBlock, newsize)
        U = IBlock(UafterBlock, newsize)
        V = IBlock(VafterBlock, newsize)
        # 将图像调至原来大小
        Y = IFill(Y, size)
        U = IFill(U, size)
        V = IFill(V, size)
        # YUV转回RGB，可根据RGB转YUV的式子反解得到这里需要的转换表达式
        R = []
        G = []
        B = []
        for i in range(0, len(Y)):
            R.append(Y[i] + 1.403 * (V[i] - 128))
            G.append(Y[i] - 0.343 * (U[i] - 128) - 0.714 * (V[i] - 128))
            B.append(Y[i] + 1.770 * (U[i] - 128))
        # 结果
        x = image.size[0]
        y = image.size[1]
        res = Image.new("RGB", (x, y))
        for i in range(0, x):
            for j in range(0, y):
                res.putpixel((i, j), (int(R[i * y + j]), int(G[i * y + j]), int(B[i * y + j])))
        if count == 1:
            res.save("animal.jpeg")
            img1 = Image.open("animal.jpeg")
            img2 = Image.open("动物照片.jpg")
            img3 = Image.open("animal.gif")
            img1 = img1.convert('L')
            img2 = img2.convert('L')
            img3 = img3.convert('L')
            print("MSE between jpeg and source picture is:{}".format(compute_mse(img1, img2)))
            print("MSE between gif and source picture is:{}".format(compute_mse(img3, img2)))
        if count == 2:
            res.save("cartoon.jpeg")
            img1 = Image.open("cartoon.jpeg")
            img2 = Image.open("动物卡通图片.jpg")
            img3 = Image.open("cartoon.gif")
            img1 = img1.convert('L')
            img2 = img2.convert('L')
            img3 = img3.convert('L')
            print("MSE between jpeg and source picture is:{}".format(compute_mse(img1, img2)))
            print("MSE between gif and source picture is:{}".format(compute_mse(img3, img2)))
        count += 1

