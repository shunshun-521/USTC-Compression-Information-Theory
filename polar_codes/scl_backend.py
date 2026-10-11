import numpy as np
import sc_backend_fn as function
import CRC as CRC

def scl_decoder(y_llr, information_pos, frozen_bit, decode_para, crc_n):
    N = y_llr.size
    n = int(np.log2(N))
    llr_matrix = np.ones((n + 1, N))
    llr_matrix[llr_matrix == 1] = float('nan')
    bit_matrix = llr_matrix.copy()
    llr_matrix[0] = y_llr
    list_max= decode_para[0]
    pm_method=decode_para[1]
    split_pos=information_pos                                               #采用传统的方法，在所有信息位分裂，未裁剪和减复杂度
    llr_list = [llr_matrix]
    bit_list = [bit_matrix]
    #print(llr_list)
    #print(bit_list)
    pm_list = [0]
    split_loc=0
    split_len=len(split_pos)
    l_now=1

    while split_len-1 >= split_loc:
        for i in range(l_now):
            llr_matrix_temp = llr_list[i]
            bit_matrix_temp = bit_list[i]
            pm_temp = pm_list[i]

            matrix_temp = sc_stepping_decoder(llr_matrix_temp, bit_matrix_temp, information_pos, frozen_bit,
                                              split_pos[split_loc])

            llr_list[i] = matrix_temp[0]
            bit_list[i] = matrix_temp[1]
            right_pm_update = function.get_pm_update(
                matrix_temp[0][n][split_pos[split_loc - 1] + 1:split_pos[split_loc] + 1],
                matrix_temp[1][n][split_pos[split_loc - 1] + 1:split_pos[split_loc] + 1], pm_method)
            pm_list[i] = pm_temp + right_pm_update
            llr_list.append(matrix_temp[0].copy())
            bit_matrix_wrong = matrix_temp[1].copy()
            bit_matrix_wrong[n][split_pos[split_loc]] = 1 - bit_matrix_wrong[n][split_pos[split_loc]]
            bit_list.append(bit_matrix_wrong.copy())
            wrong_pm_update = function.get_pm_update(
                matrix_temp[0][n][split_pos[split_loc - 1] + 1:split_pos[split_loc] + 1],
                bit_matrix_wrong[n][split_pos[split_loc - 1] + 1:split_pos[split_loc] + 1], pm_method)
            pm_list.append(pm_temp + wrong_pm_update)

        if l_now > list_max/2:
            pm_list_arg=np.argsort(pm_list)
            del_list_arg=pm_list_arg[list_max:]
            #删减多余的列表分支
            pm_list = [pm_list[i] for i in range(len(pm_list)) if i not in del_list_arg]
            llr_list = [llr_list[i] for i in range(len(llr_list)) if i not in del_list_arg]
            bit_list = [bit_list[i] for i in range(len(bit_list)) if i not in del_list_arg]
        l_now = len(pm_list)
        split_loc += 1

    if split_pos[-1] != N-1:
        for i in range(l_now):
            llr_matrix_temp = llr_list[i]
            bit_matrix_temp = bit_list[i]
            pm_temp = pm_list[i]
            matrix_temp = sc_stepping_decoder(llr_matrix_temp, bit_matrix_temp, information_pos, frozen_bit,N-1)
            llr_list[i] = matrix_temp[0]
            bit_list[i] = matrix_temp[1]
            right_pm_update = function.get_pm_update(
                matrix_temp[0][n][split_pos[split_loc - 1]+1:N],
                matrix_temp[1][n][split_pos[split_loc - 1]+1:N], pm_method)
            pm_list[i] = pm_temp + right_pm_update

    #接下来提取list中的译码值经过crc校验,采取的是排序pm，从最小的开始经过CRC，第一个通过CRC的即作为译码值输出
    pm_argsort=np.argsort(pm_list)
    for i in pm_argsort:
        u_d_temp=bit_list[i][n]
        u_d_temp = np.array([0 if u_d_temp[i] == 0 else 1 for i in range(u_d_temp.size)])
        u_d_temp_info = u_d_temp[information_pos]
        u_d_temp_info = list(u_d_temp_info)
        crc_c = CRC.CRC(u_d_temp_info, crc_n)
        flag = crc_c.detection()
        if flag == 1:
            u_d=u_d_temp
            break
        elif flag == 0 and i == pm_argsort[-1]:
            u_d_temp=bit_list[pm_argsort[0]][n]
            u_d_temp = np.array([0 if u_d_temp[i] == 0 else 1 for i in range(u_d_temp.size)])
            u_d=u_d_temp

    return u_d

def sc_stepping_decoder(llr_matrix,bit_matrix,information_pos,frozen_bit,split_pos):           #运行到判决split_pos的位置，注意是split_pos判决完成后返回的
    N = int(bit_matrix[0].size)
    n = int(np.log2(N))
    loc=function.get_up_loc(bit_matrix)
    position = [loc[0], loc[1], n, N]

    while bit_matrix[n][split_pos] != 0 and bit_matrix[n][split_pos] != 1:
        up_llr=llr_matrix[position[0]][position[1]:position[1]+2**(position[2]-position[0])]
        up_bit=bit_matrix[position[0]][position[1]:position[1]+2**(position[2]-position[0])]
        left_llr=llr_matrix[position[0]+1][position[1]:position[1]+2**(position[2]-position[0]-1)]
        left_bit=bit_matrix[position[0]+1][position[1]:position[1]+2**(position[2]-position[0]-1)]
        right_llr=llr_matrix[position[0]+1][position[1]+2**(position[2]-position[0]-1):position[1]+2**(position[2]-position[0])]
        right_bit=bit_matrix[position[0]+1][position[1]+2**(position[2]-position[0]-1):position[1]+2**(position[2]-position[0])]

        if function.all_num(up_bit) == 1:
            position = function.up(position)
        else:
            if function.all_num(right_bit) == 1:
                up_bit = function.get_up_bit(left_bit, right_bit)
                bit_matrix[position[0]][position[1]:position[1] + 2 ** (position[2] - position[0])] = up_bit.copy()
            else:
                if function.all_num(right_llr) == 1:
                    if position[0] == position[2] - 1:                  #右底
                        right_bit_pos = position[1] + 1
                        right_bit = function.get_right_bit(right_llr, information_pos, frozen_bit, right_bit_pos)
                        bit_matrix[position[0] + 1][position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (
                                    position[2] - position[0])] = right_bit
                    else:
                        position = function.rightdown(position)
                else:
                    if function.all_num(left_bit) == 1:
                        right_llr = function.get_right_llr(left_bit, up_llr)
                        llr_matrix[position[0] + 1][position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (
                                    position[2] - position[0])] = right_llr
                    else:
                        if function.all_num(left_llr) == 0:
                            left_llr = function.get_left_llr(up_llr)
                            llr_matrix[position[0] + 1][
                            position[1]:position[1] + 2 ** (position[2] - position[0] - 1)] = left_llr
                        else:
                            if position[0] == position[2] - 1:         #左底
                                left_bit_pos = position[1]
                                left_bit = function.get_left_bit(left_llr, information_pos, frozen_bit, left_bit_pos)
                                bit_matrix[position[0] + 1][
                                position[1]:position[1] + 2 ** (position[2] - position[0] - 1)] = left_bit
                            else:
                                position = function.leftdown(position)
    return [llr_matrix,bit_matrix]

def fsc_decoder(y_llr,information_pos,frozen_bit,decode_para):
    N = y_llr.size
    n = int(np.log2(N))
    node_list = function.node_identify(N,information_pos,decode_para[1])

    llr_matrix = np.ones((n + 1, N))
    llr_matrix[llr_matrix == 1] = float('nan')
    bit_matrix = llr_matrix.copy()
    llr_matrix[0] = y_llr
    position = [0, 0, n, N]
    while function.all_num(bit_matrix[n]) == 0:
        node_col = int(np.floor(position[1]/(2**(n-position[0]))))
        if node_list[position[0]][node_col] == 'nan':
            up_llr = llr_matrix[position[0]][position[1]:position[1] + 2 ** (position[2] - position[0])]
            up_bit = bit_matrix[position[0]][position[1]:position[1] + 2 ** (position[2] - position[0])]
            left_llr = llr_matrix[position[0] + 1][position[1]:position[1] + 2 ** (position[2] - position[0] - 1)]
            left_bit = bit_matrix[position[0] + 1][position[1]:position[1] + 2 ** (position[2] - position[0] - 1)]
            right_llr = llr_matrix[position[0] + 1][
                        position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (position[2] - position[0])]
            right_bit = bit_matrix[position[0] + 1][
                        position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (position[2] - position[0])]

            if function.all_num(up_bit) == 1:
                position = function.up(position)
            else:
                if function.all_num(right_bit) == 1:
                    up_bit = function.get_up_bit(left_bit, right_bit)
                    bit_matrix[position[0]][position[1]:position[1] + 2 ** (position[2] - position[0])] = up_bit.copy()
                else:
                    if function.all_num(right_llr) == 1:
                        if position[0] == position[2] - 1:
                            right_bit_pos = position[1] + 1
                            right_bit = function.get_right_bit(right_llr, information_pos, frozen_bit, right_bit_pos)
                            bit_matrix[position[0] + 1][
                            position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (
                                        position[2] - position[0])] = right_bit
                        else:
                            position = function.rightdown(position)
                    else:
                        if function.all_num(left_bit) == 1:
                            right_llr = function.get_right_llr(left_bit, up_llr)
                            llr_matrix[position[0] + 1][
                            position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (
                                        position[2] - position[0])] = right_llr
                        else:
                            if function.all_num(left_llr) == 0:
                                left_llr = function.get_left_llr(up_llr)
                                llr_matrix[position[0] + 1][
                                position[1]:position[1] + 2 ** (position[2] - position[0] - 1)] = left_llr
                            else:
                                if position[0] == position[2] - 1:
                                    left_bit_pos = position[1]
                                    left_bit = function.get_left_bit(left_llr, information_pos, frozen_bit, left_bit_pos)
                                    bit_matrix[position[0] + 1][
                                    position[1]:position[1] + 2 ** (position[2] - position[0] - 1)] = left_bit
                                else:
                                    position = function.leftdown(position)

        else:
            up_llr = llr_matrix[position[0]][position[1]:position[1] + 2 ** (position[2] - position[0])]
            node_result = function.node_process(up_llr,node_list[position[0]][node_col])
            bit_matrix[position[0]][position[1]:position[1] + 2 ** (position[2] - position[0])] = node_result[0]
            bit_matrix[n][position[1]:position[1] + 2 ** (position[2] - position[0])] = node_result[1]
            position = function.up(position)
    u_d = bit_matrix[n]
    return u_d






def sc_flip1_decoder(y_llr,information_pos,frozen_bit,flip_pos):
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
                        right_bit=function.get_right_bit_flip1(right_llr,information_pos,frozen_bit,right_bit_pos,flip_pos)
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
                                left_bit=function.get_left_bit_flip1(left_llr,information_pos,frozen_bit,left_bit_pos,flip_pos)
                                bit_matrix[position[0]+1][position[1]:position[1]+2**(position[2]-position[0]-1)]=left_bit
                            else:
                                position = function.leftdown(position)

    u_d=[bit_matrix[n],llr_matrix[n]]
    return u_d

