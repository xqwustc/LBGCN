import tensorflow as tf

user_embedding = [[1, 2, 3],
                  [5, 6, 7],
                  [3, 1, 5],
                  [8, 4, 2]]

item_embedding = [[5, 2, 7],
                  [3, 7, 2],
                  [2, 5, 2],
                  [9, 3, 1]]

ego_embeddings = tf.concat([user_embedding, item_embedding], axis=0)
print(ego_embeddings)

all_embeddings = [ego_embeddings]
print('all_embeddings:', all_embeddings)

for k in range(0, 3):  # 3 层嵌入

    A = [[0, 0, 0, 0, 1, 0, 0, 1],
         [0, 0, 0, 0, 0, 1, 0, 1],
         [0, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 1, 0, 1, 0],
         [1, 0, 0, 1, 0, 0, 0, 0],
         [0, 1, 1, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 1, 0, 0, 0, 0]]

    side_embeddings =tf.matmul(A,ego_embeddings)
    # print('第'+str(k+1)+'次的embeddings:',side_embeddings)

    ego_embeddings = side_embeddings  # 结果是个张量
    print('ego_embeddings = side_embedding后的ego_embeddings:',ego_embeddings)
    all_embeddings += [ego_embeddings]  #  把多个张量拼接在一起
    print('第' + str(k + 1) + '次all_embeddings += [ego_embeddings]后的all_embeddings:', all_embeddings)


all_embeddings = tf.stack(all_embeddings, 1)  # 把多个array重新变回多个张量放在一起，然后做平均操作
print('tf.stack(all_embeddings, 1)后的all_embeddings:',all_embeddings)

all_embeddings = tf.reduce_mean(all_embeddings, axis=1, keepdims=True)  # 在[a,b,c]   (a+b+c)/3 列上求平均
print('tf.reduce_mean(all_embeddings, axis=1, keepdims=True)后的all_embeddings:',all_embeddings)

u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [4, 4], 0)
print('u_g_embeddings:',u_g_embeddings)
print('i_g_embeddings:',i_g_embeddings)

