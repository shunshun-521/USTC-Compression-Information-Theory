import numpy as np
import sc_backend_fn as function
from encoder import polar_encode

def bp_decoder(y_llr,information_pos,frozen_bit,decode_para,crc_n):
    N = y_llr.size
    n = int(np.log2(N))

    if decode_para[1] == 'max_iter':

        bp_max_iter=int(decode_para[0])
        #初始化左传播和右传播矩阵
        left_matrix=np.zeros([N,n+1])
        right_matrix=np.zeros([N,n+1])
        left_matrix[:,n]=y_llr
        temp_value=(1-2*frozen_bit)*np.infty
        temp=[temp_value if i not in information_pos else 0 for i in range(N)]
        right_matrix[:,0]=temp

        for iter in range(bp_max_iter):
            for i in range(n):
                left_matrix[:,n-i-1]=function.bp_update_left(left_matrix[:,n-i],right_matrix[:,n-i-1],n-i)

            for i in range(n):
                right_matrix[:,i+1]=function.bp_update_right(left_matrix[:,i+1],right_matrix[:,i],i+1)


        u_d_llr=left_matrix[:,0]+right_matrix[:,0]
        u_d=[0 if u_d_llr[i] >= 0 else 1 for i in range(N)]
        u_d=np.array(u_d)


    elif decode_para[1] == 'g_matrix':

        # 初始化左传播和右传播矩阵
        left_matrix = np.zeros([N, n + 1])
        right_matrix = np.zeros([N, n + 1])
        left_matrix[:, n] = y_llr
        temp_value = (1 - 2 * frozen_bit) * np.infty
        temp = [temp_value if i not in information_pos else 0 for i in range(N)]
        right_matrix[:, 0] = temp
        flag=0
        iter_num=1

        while flag == 0 and iter_num <= decode_para[0]:
            for i in range(n):
                left_matrix[:, n - i - 1] = function.bp_update_left(left_matrix[:, n - i], right_matrix[:, n - i - 1],n - i)

            for i in range(n):
                right_matrix[:, i + 1] = function.bp_update_right(left_matrix[:, i + 1], right_matrix[:, i], i + 1)

            u_d_llr = left_matrix[:, 0] + right_matrix[:, 0]
            u_d = [0 if u_d_llr[i] >= 0 else 1 for i in range(N)]
            x_d_llr = left_matrix[:, n] + right_matrix[:, n]
            x_d = [0 if x_d_llr[i] >= 0 else 1 for i in range(N)]
            G = function.generate_matrix(n)
            x_g = u_d * G
            x_g = np.array(x_g % 2)
            #print(x_g)
            cor = [0 if x_g[0][i]==x_d[i] else 1 for i in range(N)]
            if sum(cor) == 0:
                flag = 1
            else:
                flag = 0
            iter_num += 1
        u_d = np.array(u_d)


    elif decode_para[1] == 'crc_es':

        # 初始化左传播和右传播矩阵
        left_matrix = np.zeros([N, n + 1])
        right_matrix = np.zeros([N, n + 1])
        left_matrix[:, n] = y_llr
        temp_value = (1 - 2 * frozen_bit) * np.infty
        temp = [temp_value if i not in information_pos else 0 for i in range(N)]
        right_matrix[:, 0] = temp
        flag = 0
        iter_num = 1

        while flag == 0 and iter_num <= decode_para[0]:
            for i in range(n):
                left_matrix[:, n - i - 1] = function.bp_update_left(left_matrix[:, n - i], right_matrix[:, n - i - 1],n - i)

            for i in range(n):
                right_matrix[:, i + 1] = function.bp_update_right(left_matrix[:, i + 1], right_matrix[:, i], i + 1)

            u_d_llr = left_matrix[:, 0] + right_matrix[:, 0]
            u_d = np.array([0 if u_d_llr[i] >= 0 else 1 for i in range(N)])
            u_d_info = u_d[information_pos]
            u_d_info = list(u_d_info)
            crc_c=CRC.CRC(u_d_info,crc_n)
            flag=crc_c.detection()
            # if flag == 1:
            #     print('It work!',iter_num)

            iter_num += 1
        u_d = np.array(u_d)
    else:
        print('This is not a early stopping method of BP decoder or s have not added it to the endecoder.bp_decoder !')
        quit()
    return u_d













