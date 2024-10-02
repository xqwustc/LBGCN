import tensorflow as tf
user_embedding = [[1, 2, 3],
                  [5, 6, 7],
                  [3, 1, 5],
                  [8, 4, 2]]
# all_embeddings = tf.stack(user_embedding, 1)
print(user_embedding)
all_embeddings = tf.reduce_mean(user_embedding, axis=0, keepdims=True)
print(all_embeddings)





