import numpy as np
import matplotlib.pyplot as plt
import math
import random
import logging
logging.getLogger().setLevel(logging.INFO)


def show_state(x, y, angle):
    plt.clf()
    plt.plot([-3, 3], [-1, -1], linestyle='-', color='b')
    plt.plot([-3, 3], [1, 1], linestyle='-', color='b')
    plt.plot([-3, -3], [-1, 1], linestyle='-', color='b')
    plt.plot([3, 3], [-1, 1], linestyle='--', color='g')
    agent = plt.Circle((x, y), radius=0.2, color='r', fill=False)
    plt.gca().add_patch(agent)
    plt.plot([x, x + math.cos(angle) * 0.4], [y, y + math.sin(angle) * 0.4], linestyle='-', color='r')
    plt.axis('scaled')
    plt.axis([-4, 4, -1.2, 1.2])
    plt.axis('off')
    plt.show()
    plt.pause(0.001)


def update_state(x, y, angle, velocity, angle_speed):
    angle += angle_speed
    x += velocity * math.cos(angle)
    y += velocity * math.sin(angle)
    return x, y, angle


def state_is_end(x, y, angle):
    if x < -3 or x > 3 or y < -1 or y > 1:
        return True
    else:
        return False


def state_is_finished(x, y, angle):
    if x > 3 and -1 < y < 1:
        return True
    else:
        return False


def get_reward(x, y, angle, velocity):
    reward = velocity * math.cos(angle)
    return reward


def state_to_feature(x, y, angle):
    feature = np.array([y, math.cos(angle), math.sin(angle), 1])  # Last dimension is for bias
    return feature


def action_to_physics(action):
    velocity = 0.1
    if action == 0:
        angle_speed = math.pi * 0.1
    else:
        angle_speed = - math.pi * 0.1
    return velocity, angle_speed


def initialize_param(feature_dim, output_dim):
    weight = np.random.normal(0, 1.0, (feature_dim, output_dim))
    return weight


def get_q_value(feature, weight):
    q_value = np.dot(feature, weight)
    return q_value


def get_max_q_action(q_value):
    max_q_value = q_value.max()
    max_action = q_value.argmax()
    return max_q_value, max_action


def explore_exploit(action, exploration_rate):
    if random.random() < exploration_rate:  # Exploration
        if random.random() < 1.0 / 2:
            new_action = 0
        else:
            new_action = 1
    else:  # Exploitation
        new_action = action
    return new_action


def update_weight(weight, delta_weight, feature, q_value, action, gt_q_value):
    momentum = 0.9
    learning_rate = 0.001
    weight_decay = 0.001
    delta = q_value[action] - gt_q_value
    gradient = delta * feature + weight_decay * weight[:, action]
    delta_weight[:, action] = momentum * delta_weight[:, action] + learning_rate * gradient
    weight[:, action] = weight[:, action] - delta_weight[:, action]
    return weight, delta_weight


def compute_loss(q_value, action, gt_q_value):
    loss = (q_value[action] - gt_q_value) ** 2
    return loss


def random_state():
    x = random.random() * 6 - 3
    y = random.random() * 2 - 1
    angle = random.random() * 2 * math.pi - math.pi
    return x, y, angle


def q_learn():
    plt.ion()
    train_iter = 100  # Number of training iterations
    exploration_rate = np.linspace(1.0, 0.1, train_iter)
    discount_factor = 0.9  # Discounted factor
    weight = initialize_param(4, 2)  # Initialize weights
    delta_weight = np.zeros(weight.shape)
    # for i in range(train_iter):
    i = 0
    while True:
        i += 1
        x, y, angle = random_state()
        cumulative_reward, discount = 0, 1
        while True:
            show_state(x, y, angle)
            feature = state_to_feature(x, y, angle)
            q_value = get_q_value(feature, weight)
            max_q_value, max_action = get_max_q_action(q_value)
            if i >= train_iter:
                explore_rate = 0.1
            else:
                explore_rate = exploration_rate[i]
            action = explore_exploit(max_action, explore_rate)
            velocity, angle_speed = action_to_physics(action)
            x, y, angle = update_state(x, y, angle, velocity, angle_speed)
            if state_is_finished(x, y, angle):
                reward = 1
                next_max_q_value = 0
            elif state_is_end(x, y, angle):
                reward = -10
                next_max_q_value = 0
            else:
                reward = get_reward(x, y, angle, velocity)
                next_feature = state_to_feature(x, y, angle)
                next_q_value = get_q_value(next_feature, weight)
                next_max_q_value, next_max_action = get_max_q_action(next_q_value)
            cumulative_reward += discount * reward
            discount *= discount_factor
            gt_q_value = reward + discount_factor * next_max_q_value
            loss = compute_loss(q_value, action, gt_q_value)
            weight, delta_weight = update_weight(weight, delta_weight, feature, q_value, action, gt_q_value)
            if state_is_end(x, y, angle):
                break
        logging.info('iteration: %d, exploration rate: %.4f, cumulative reward: %.4f',
                     i, explore_rate, cumulative_reward)
    plt.ioff()
    plt.show()
    return weight


if __name__ == '__main__':
    w = q_learn()