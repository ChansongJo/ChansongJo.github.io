# Contextual Bandit problem

```python
# context 멀티암드밴딧 구현
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np


class contextual_bandit():
    def __init__(self):
        self.state = 0
        self.bandits = np.array([[0.2, 0, -0.0, -5],
                                 [0.1, -5, 1, 0.25],
                                 [-5, 5, 5, 5]])
        self.answer = self.bandits.argmin(axis=1)
        self.num_bandits, self.num_actions = self.bandits.shape  # 3, 4

    def getBandit(self):
        self.state = np.random.randint(0, len(self.bandits))
        return self.state

    def pullArm(self, action):
        bandit = self.bandits[self.state, action]
        result = np.random.randn(1)
        if result > bandit:
            return 1
        else:
            return -1


class agent():
    def __init__(self, lr, s_size, a_size):
        self.state_in = tf.placeholder(shape=[1], dtype=tf.int32)
        state_in_OH = slim.one_hot_encoding(self.state_in, s_size)
        ouput = slim.fully_connected(state_in_OH, a_size,
                                     biases_initializer=None,
                                     activation_fn=tf.nn.sigmoid,
                                     weights_initializer=tf.ones_initializer())
        self.output = tf.reshape(ouput, [-1])
        self.chosen_action = tf.argmax(self.output, 0)
        self.reward_holder = tf.placeholder(shape=[1], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[1], dtype=tf.int32)
        self.responsible_weight = tf.slice(self.output, self.action_holder, [1])
        self.loss = -(tf.log(self.responsible_weight) * self.reward_holder)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
        self.update = optimizer.minimize((self.loss))


# training start
cBandit = contextual_bandit()
myAgent = agent(lr=0.001, s_size=cBandit.num_bandits, a_size=cBandit.num_actions)
weights = tf.trainable_variables()[0]

total_episodes = 10000
total_reward = np.zeros([cBandit.num_bandits, cBandit.num_actions])
e = 0.1  # epsilon

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    for i in range(total_episodes):
        s = cBandit.getBandit()
        if np.random.rand(1) < e:
            action = np.random.randint(cBandit.num_actions)
        else:
            action = sess.run(myAgent.chosen_action, feed_dict={myAgent.state_in: [s]})

        reward = cBandit.pullArm(action)
        _, ww = sess.run([myAgent.update, weights], feed_dict={myAgent.reward_holder: [reward],
                                                               myAgent.action_holder: [action],
                                                               myAgent.state_in: [s]})

        total_reward[s, action] += reward
        if i % 500 == 0:
            print("Mean reward for each of the {} bandits: {}".format(cBandit.num_bandits, np.mean(total_reward, axis=1)))


for a in range(cBandit.num_bandits):
    print("The agent thinks action {} is ths most promising for bandit {}".format((np.argmax(ww[a])+1), (a+1)))
    print('curr_weights: ', ww[a])
    if np.argmax(ww[a]) == np.argmin(cBandit.bandits[a]):
        print("correct!!")
    else:
        print("wrong")
```

* output

```text
Mean reward for each of the 3 bandits: [-0.25  0.    0.  ]
Mean reward for each of the 3 bandits: [ 1.75 32.   34.  ]
Mean reward for each of the 3 bandits: [42.   71.5  68.25]
Mean reward for each of the 3 bandits: [ 80.25 106.5  107.  ]
Mean reward for each of the 3 bandits: [122.5  141.75 145.5 ]
Mean reward for each of the 3 bandits: [161.25 177.75 179.75]
Mean reward for each of the 3 bandits: [205.25 215.5  213.5 ]
Mean reward for each of the 3 bandits: [243.25 258.25 245.75]
Mean reward for each of the 3 bandits: [279.5  298.   284.25]
Mean reward for each of the 3 bandits: [317.25 336.   321.  ]
Mean reward for each of the 3 bandits: [352.5  377.75 357.  ]
Mean reward for each of the 3 bandits: [394.75 411.5  393.5 ]
Mean reward for each of the 3 bandits: [431.   447.75 429.  ]
Mean reward for each of the 3 bandits: [471.75 482.25 458.75]
Mean reward for each of the 3 bandits: [504.5  521.25 498.5 ]
Mean reward for each of the 3 bandits: [543.   561.25 535.  ]
Mean reward for each of the 3 bandits: [577.   601.25 570.5 ]
Mean reward for each of the 3 bandits: [619.   639.   604.25]
Mean reward for each of the 3 bandits: [656.   674.75 641.  ]
Mean reward for each of the 3 bandits: [688.25 714.   676.5 ]
The agent thinks action 4 is ths most promising for bandit 1
curr_weights:  [0.9938025 0.9981178 0.9997381 1.6165975]
correct!!
The agent thinks action 2 is ths most promising for bandit 2
curr_weights:  [0.9989254 1.639084  0.9848577 0.9983877]
correct!!
The agent thinks action 1 is ths most promising for bandit 3
curr_weights:  [1.6429801  0.9769494  0.97940844 0.9785893 ]
correct!!
```
