import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

digits = load_digits()
X = digits.data
y_ = digits.target
y = LabelBinarizer().fit_transform(y_)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

def add_layer(inputs,in_size,out_size,layer_name,activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size])+0.1)
    output_raw = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        output = output_raw
    else:
        output = activation_function(output_raw)
    tf.summary.histogram(layer_name + '/output', output)
    return output

xs = tf.placeholder(tf.float32, [None, 64])
ys = tf.placeholder(tf.float32, [None, 10])

l1 = add_layer(xs, 64, 50, 'L1', activation_function = tf.nn.tanh)
prep = add_layer(l1, 50, 10, 'L2', activation_function = tf.nn.softmax)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prep), reduction_indices=[1]))
tf.summary.scalar('loss', cross_entropy)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()
merged = tf.summary.merge_all()

train_writer = tf.summary.FileWriter('logs/train', sess.graph)
test_writer = tf.summary.FileWriter('logs/test', sess.graph)

init = tf.global_variables_initializer()
sess.run(init)

for i in range(2000):
    sess.run(train_step, feed_dict={xs:X_train, ys:y_train})
    if i % 50 == 0:
        train_result = sess.run(merged, feed_dict = {xs:X_train, ys:y_train})
        test_result = sess.run(merged, feed_dict = {xs:X_test, ys:y_test})
        train_writer.add_summary(train_result, i)
        test_writer.add_summary(test_result, i)