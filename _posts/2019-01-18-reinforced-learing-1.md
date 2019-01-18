# 강화학습 소개

> 강화 학습은 상호 관계에 바탕을 두고 진행 된다. agent를 둘러싼 action과 world의 상호작용을 통해 agent를 학습

강화 학습의 보상함수는 결국 reward function을 어떻게 진행하느냐에 달려 있다. 할인된 기대 보상의 최적화를 통해 에이전트가 최상의 선택을 하도록 유도한다

# Bandit Problem

> 강화 학습의 가장 단순한 문제! 손잡이가 n개인 슬롯머신 문제

agent 가 여러개의 bandit 머신 중 어떤 방식으로 액션을 해야 최고의 보상을 얻을 수 있는지를 학습하는 문제.

1. 액션 의존성: 각 액션이 다른 보상을 가져온다 A: 3, B: 4 -> B를 선택
2. 시간 의존성: 액션과 주어지는 보상의 시간이 다르다. 어떠한 행위가 종료되어야만 그것이 옳은 선택이었다는 것을 알 수 있다. -> deep_dialog reward
3. 상태 의존성: 어떤 액션에 대한 보상은 환경에 의해 좌우 된다. 개별 시도 마다 환경은 달라지고 각 행위에 따라 변화된다.

> 이때 agent가 취하는 일련의 액션을 정책(policy)라고 칭하며 주어진 환경 내에서 최대의 보상을 얻는 정책을 최적의 정책으로 간주하게 된다.

## 정책 경사 - Policy Gradient

* epsilon greedy policy -> epsilon의 확률로 action을 번갈아 선택하는 알고리즘
* 테스트할 환경에서는 정답 1, 오답 -1의 보상을 받는다
* 정책 손실 함수는 \\(Loss = -log(\pi)*A\\) 를 이용한다. (\\(\pi\\)는 정책을 의미)

```python
# 기초 멀티암드밴딧 구현
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
# writing down the list of the bandit_arms
bandit_arms = [0.2, 0, -0.2, -0.3]
num_arms = len(bandit_arms)

def pullBandit(bandit):
    result = np.random.randn(1)
    if result > bandit:
        return 1
    else:
        return -1

# params
rate = 1e-3
episodes = 1000

# 초기 기대치 [1 1 1 1]
w = tf.Variable(tf.ones([num_arms]))
output = tf.nn.softmax(w)

reward_holder = tf.placeholder(shape=[1], dtype = tf.float32)
action_holder = tf.placeholder(shape=[1], dtype= tf.int32)

# slice?
responsible_ouput = tf.slice(output, action_holder, [1])
loss = -(tf.log(responsible_ouput)*reward_holder)

optimizer = tf.train.AdamOptimizer(learning_rate=rate)
update = optimizer.minimize(loss)

# 손잡이 마다의 점수판
total_reward = np.zeros(num_arms)

init = tf.global_variables_initializer()
# training
with tf.Session() as sess:
    sess.run(init)
    
    for i in range(episodes):
        actions = sess.run(output)
        action = np.argmax(actions ==  np.random.choice(actions, p=actions))
        
        reward = pullBandit(bandit_arms[action])
        
        _, resp, ww = sess.run([update, responsible_ouput, w],
                                            feed_dict = {reward_holder: [reward], action_holder: [action]})
        
        total_reward[action] += reward
        if i % 50 == 0:
            print("Running reward for the {} arms of the bandit: {}".format(num_arms, total_reward))
            
        if i % 201 == 0:
            print()
            print("중간 점검")
            print('current_weight', ww)
            print("\nThe agent thinks arm %s is the most promising ....." % (np.argmax(ww)+1))
            if np.argmax(ww) == np.argmax(-np.array(bandit_arms)):
                print("... and it was right!")
            else:
                print("... and it was wrong!")

print("\nThe agent thinks arm %s is the most promising ....." % (np.argmax(ww)+1))
if np.argmax(ww) == np.argmax(-np.array(bandit_arms)):
    print("... and it was right!")
else:
    print("... and it was wrong!")
```

### output
```
Running reward for the 4 arms of the bandit: [-28.   9.  41.  55.]

중간 점검
current_weight [1.001 0.999 0.999 0.999]

The agent thinks arm 1 is the most promising .....
... and it was wrong!
Running reward for the 4 arms of the bandit: [-34.   9.  45.  55.]
Running reward for the 4 arms of the bandit: [-38.  14.  49.  62.]
Running reward for the 4 arms of the bandit: [-40.  19.  53.  67.]
Running reward for the 4 arms of the bandit: [-37.  21.  53.  74.]

중간 점검
current_weight [0.9615943 1.0027988 1.0093577 1.0268886]

The agent thinks arm 4 is the most promising .....
... and it was right!
Running reward for the 4 arms of the bandit: [-37.  23.  52.  81.]
Running reward for the 4 arms of the bandit: [-35.  20.  51.  81.]
Running reward for the 4 arms of the bandit: [-32.  18.  51.  84.]
Running reward for the 4 arms of the bandit: [-32.  14.  52.  89.]

중간 점검
current_weight [0.96798825 0.98290676 0.99810076 1.0516318 ]

The agent thinks arm 4 is the most promising .....
... and it was right!
Running reward for the 4 arms of the bandit: [-23.  17.  56.  91.]
Running reward for the 4 arms of the bandit: [-27.  16.  62.  94.]
Running reward for the 4 arms of the bandit: [-34.  15.  59. 107.]
Running reward for the 4 arms of the bandit: [-40.  15.  62. 110.]

중간 점검
current_weight [0.93602943 0.96945286 1.0057229  1.0875003 ]

The agent thinks arm 4 is the most promising .....
... and it was right!
Running reward for the 4 arms of the bandit: [-36.  14.  65. 116.]
Running reward for the 4 arms of the bandit: [-35.  12.  71. 123.]
Running reward for the 4 arms of the bandit: [-39.  15.  74. 127.]
Running reward for the 4 arms of the bandit: [-46.  15.  69. 129.]

중간 점검
current_weight [0.9119518  0.95824987 1.0122811  1.1140939 ]

The agent thinks arm 4 is the most promising .....
... and it was right!
Running reward for the 4 arms of the bandit: [-50.  14.  70. 129.]
Running reward for the 4 arms of the bandit: [-54.  15.  74. 136.]
Running reward for the 4 arms of the bandit: [-56.   9.  73. 141.]

The agent thinks arm 4 is the most promising .....
... and it was right
```