

class FreNum:
    def __init__(self):
        self.index = -1
        self.num = -1
        self.fre = 0

    def init(self, index, num, fre):
        self.index = index # 第一次出现的位置
        self.num = num
        self.fre = fre

    def update(self):
        self.fre += 1

def topk(nums):
    freNum_dict = {}

    for index, num in enumerate(nums):
        if freNum_dict.get(num) is None:
            freNum = FreNum()
            freNum.init(index, num, 1)
            freNum_dict[num] = freNum
        else:
            freNum = freNum_dict[num]
            freNum.update()
            freNum_dict[num] = freNum


    sorted_list = sorted(freNum_dict.values(), key=lambda item: (item.fre), reverse=True)

    sorted_list = sorted(sorted_list, key=lambda item: (item.index), reverse=False)


    res = sorted_list[0].num + sorted_list[1].num

    # return res
    print(res)
    return

if __name__ == '__main__':
    topk([1, 1, 1, 2, 2, 3])