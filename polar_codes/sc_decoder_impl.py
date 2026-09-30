import numpy as np
import sc_functions as function
from sc_functions import *

def sc_decoder(y_llr,information_pos,frozen_bit):
    N = y_llr.size
    #print(N)
    n = int(np.log2(N))
    llr_matrix=np.ones((n+1,N))
    llr_matrix[llr_matrix == 1] = float('nan')
    bit_matrix=llr_matrix.copy()
    llr_matrix[0]=y_llr
    position=[0,0,n,N]
    while function.all_num(bit_matrix[n]) == 0:
        up_llr=llr_matrix[position[0]][position[1]:position[1]+2**(position[2]-position[0])]
        up_bit=bit_matrix[position[0]][position[1]:position[1]+2**(position[2]-position[0])]
        left_llr=llr_matrix[position[0]+1][position[1]:position[1]+2**(position[2]-position[0]-1)]
        left_bit=bit_matrix[position[0]+1][position[1]:position[1]+2**(position[2]-position[0]-1)]
        right_llr=llr_matrix[position[0]+1][position[1]+2**(position[2]-position[0]-1):position[1]+2**(position[2]-position[0])]
        right_bit=bit_matrix[position[0]+1][position[1]+2**(position[2]-position[0]-1):position[1]+2**(position[2]-position[0])]
        #print(left_llr)

        if function.all_num(up_bit) == 1:
            position=function.up(position)
        else:
            if function.all_num(right_bit) == 1:
                up_bit=function.get_up_bit(left_bit,right_bit)
                bit_matrix[position[0]][position[1]:position[1] + 2 ** (position[2] - position[0])]=up_bit.copy()
            else:
                if function.all_num(right_llr) == 1:
                    if position[0] == position[2]-1:
                        right_bit_pos=position[1]+1
                        right_bit=function.get_right_bit(right_llr,information_pos,frozen_bit,right_bit_pos)
                        bit_matrix[position[0]+1][position[1]+2**(position[2]-position[0]-1):position[1]+2**(position[2]-position[0])]=right_bit
                    else:
                        position=function.rightdown(position)
                else:
                    if function.all_num(left_bit) == 1:
                        right_llr=function.get_right_llr(left_bit,up_llr)
                        llr_matrix[position[0]+1][position[1]+2**(position[2]-position[0]-1):position[1]+2**(position[2]-position[0])]=right_llr
                    else:
                        if function.all_num(left_llr) == 0:
                            left_llr = function.get_left_llr(up_llr)
                            llr_matrix[position[0] + 1][position[1]:position[1] + 2 ** (position[2] - position[0] - 1)] = left_llr
                        else:
                            if position[0] == position[2]-1:
                                left_bit_pos=position[1]
                                left_bit=function.get_left_bit(left_llr,information_pos,frozen_bit,left_bit_pos)
                                bit_matrix[position[0]+1][position[1]:position[1]+2**(position[2]-position[0]-1)]=left_bit
                            else:
                                position = function.leftdown(position)

    u_d=[bit_matrix[n],llr_matrix[n]]
    return u_d

