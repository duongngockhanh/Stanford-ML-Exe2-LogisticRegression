import os
import numpy as np
from matplotlib import pyplot
from scipy import optimize
import utils

a = np.array([[[1,1,1,1],[2,2,2,2],[3,3,3,3]],[[4,4,4,4],[5,5,5,5],[6,6,6,6]]])
gom_cot = np.sum(a, axis=1, keepdims=True) #(2,1,4)
gom_tang = np.sum(gom_cot, axis=2, keepdims=True) #(2,1,1)
print(gom_tang)

'''
keepdims=True nghĩa là dữ lại số chiều của array
Ví dụ: Ở phần gom_cot, nếu không có keepdims=True thì shape = (2,4)
Điều này sẽ khiến việc kiểm soát khó hơn chút.
'''

'''
Dưới đây, ta có ma trận dạng (2x3x4)
1 1 1 1     4 4 4 4
2 2 2 2     5 5 5 5
3 3 3 3     6 6 6 6
Công thức là, (axis bằng bnh) thì (xóa chỉ số axis đó là ta thu được dạng của tổng)
+ Với axis=0, ta thu được tổng dạng (3x4)
[[5 5 5 5]
 [7 7 7 7]
 [9 9 9 9]]
+ Với axis=1, ta thu được tổng dạng (2x4)
[[ 6  6  6  6]
 [15 15 15 15]]
+ Với axis=2, ta thu được tổng dạng (2x3)
[[ 4  8 12]
 [16 20 24]]
'''
