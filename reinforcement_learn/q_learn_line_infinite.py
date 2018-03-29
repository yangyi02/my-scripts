import numpy as np
import matplotlib.pyplot as plt
import math
import random
import logging

logging.getLogger().setLevel(logging.INFO)


def show_state(x, y, angle):
    plt.clf()
    plt.plot([-5, 5], [-1, -1], linestyle='-', color='b')
    plt.plot([-5, 5], [1, 1], linestyle='-', color='b')
    plt.plot([-5, -5], [-1, 1], linestyle='-', color='b')
    plt.plot([5, 5], [-1, 1], linestyle='--', color='b')
    plt.plot([-5, 5], [0, 0], linestyle='-', color='g')
    agent = plt.Circle((x, y), radius=0.1, color='r', fill=False)
    plt.gca().add_patch(agent)
    plt.plot([x, x + math.cos(angle) * 0.2], [y, y + math.sin(angle) * 0.2], linestyle='-', color='r')
    plt.axis([-6, 6, -1.5, 1.5])
    plt.axis('equal')
    plt.show()
    plt.pause(0.001)


def test_show_state():
    plt.ioff()
    x, y, angle = 0, 0, 0
    show_state(x, y, angle)
    x, y, angle = -1, 0, -math.pi / 2
    show_state(x, y, angle)


def update_state(x, y, angle, velocity, angle_speed, time_step):
    angle += angle_speed * time_step
    x += velocity * math.cos(angle) * time_step
    y += velocity * math.sin(angle) * time_step
    return x, y, angle


def state_is_end(x, y, angle):
    if x < -5 or x > 5 or y < -1 or y > 1:
        return True
    else:
        return False


def get_simple_action():
    velocity = 0.1
    angle_speed = math.pi * 0.1
    return velocity, angle_speed


def test_simple_simulation():
    plt.ion()
    time_step = 1  # Discrete approximation time step for the continuous game
    x, y, angle = -4, 0, 0  # Robot state
    for i in range(10):
        show_state(x, y, angle)
        velocity, angle_speed = get_simple_action()
        x, y, angle = update_state(x, y, angle, velocity, angle_speed, time_step)
        if state_is_end(x, y, angle):
            break
    plt.ioff()
    plt.show()


def get_reward(x, y, angle, velocity, time_step):
    sin_global_angle = 0
    cos_global_angle = 1
    move_distance = velocity * time_step
    reward_speed = move_distance * (cos_global_angle * math.cos(angle) + sin_global_angle * math.sin(angle))
    reward_dist = - abs(y)
    reward = reward_speed + reward_dist * 0
    # logging.info('reward_speed: %.4f, reward_dist: %.4f, reward: %.4f', reward_speed, reward_dist, reward)
    return reward


def test_get_reward():
    x, y, angle, velocity, time_step = -4, 0, 0, 0.1, 1
    get_reward(x, y, angle, velocity, time_step)


def state_to_feature(x, y, angle):
    dist_to_center = y
    sin_global_angle = 0
    cos_global_angle = 1
    dist_to_angle1 = sin_global_angle * math.cos(angle) - cos_global_angle * math.sin(angle)
    dist_to_angle2 = cos_global_angle * math.cos(angle) + sin_global_angle * math.sin(angle)
    # logging.info('dist_to_center: %.4f, dist_to_angle: %.4f, %.4f', dist_to_center, dist_to_angle1, dist_to_angle2)
    feature = np.array([dist_to_center, dist_to_angle1, dist_to_angle2, 1])  # Last dimension is for bias
    return feature


def test_state_to_feature():
    x, y, angle = -4, 0, 0
    state_to_feature(x, y, angle)


def initialize_param(feature_dim, output_dim):
    weight = np.random.normal(0, 1.0, (feature_dim, output_dim))
    return weight


def test_initialize_param():
    weight = initialize_param(3 + 1, 2)
    print weight


def get_q_value(feature, weight):
    q_value = np.dot(feature, weight)
    # logging.info('q_value: [%.4f, %.4f]', q_value[0], q_value[1])
    return q_value


def test_get_q_value():
    x, y, angle = -4, 0, 0
    feature = state_to_feature(x, y, angle)
    weight = initialize_param(3 + 1, 2)
    q_value = get_q_value(feature, weight)


def get_max_q_action(q_value):
    max_q_value = q_value.max()
    max_action = q_value.argmax()
    # logging.info('max_q_value: %.4f, max_action: %d', max_q_value, max_action)
    return max_q_value, max_action


def test_get_max_q_action():
    x, y, angle = -4, 0, 0
    feature = state_to_feature(x, y, angle)
    weight = initialize_param(3 + 1, 2)
    q_value = get_q_value(feature, weight)
    max_q_value, max_action = get_max_q_action(q_value)


def action_to_physics(action):
    velocity = 0.1
    if action == 0:
        angle_speed = math.pi * 0.1
    else:
        angle_speed = - math.pi * 0.1
    # logging.info('velocity: %.4f, angle_speed: %.4f', velocity, angle_speed)
    return velocity, angle_speed


def test_action_to_physics():
    action_to_physics(0)


def q_simulation():
    plt.ion()
    time_step = 1  # Discrete approximation time step for the continuous game
    discount_factor = 0.9  # Discounted factor
    for iteration in range(5):
        logging.info('starting iteration %d', iteration)
        x, y, angle = -4, 0, 0  # Robot state
        weight = initialize_param(3 + 1, 2)
        cumulative_reward, discount = 0, 1
        for i in range(10):
            show_state(x, y, angle)
            feature = state_to_feature(x, y, angle)
            q_value = get_q_value(feature, weight)
            max_q_value, max_action = get_max_q_action(q_value)
            velocity, angle_speed = action_to_physics(max_action)
            x, y, angle = update_state(x, y, angle, velocity, angle_speed, time_step)
            reward = get_reward(x, y, angle, velocity, time_step)
            cumulative_reward += discount * reward
            discount *= discount_factor
            logging.info('cumulative reward: %.4f', cumulative_reward)
            if state_is_end(x, y, angle):
                break
    plt.ioff()
    plt.show()


def test_q_simulation():
    q_simulation()


def explore_exploit(action, exploration_rate):
    if random.random() < exploration_rate:  # Exploration
        if random.random() < 1.0 / 2:
            new_action = 0
        else:
            new_action = 1
    else:  # Exploitation
        new_action = action
    # logging.info('action: %.4f, action after exploration: %.4f', action, new_action)
    return new_action


def q_explore_simulation():
    plt.ion()
    time_step = 1  # Discrete approximation time step for the continuous game
    exploration_rate = 0.5  # Exploration rate
    discount_factor = 0.9  # Discounted factor
    for iteration in range(5):
        logging.info('starting iteration %d', iteration)
        x, y, angle = -4, 0, 0  # Robot state
        weight = initialize_param(3 + 1, 2)
        cumulative_reward, discount = 0, 1
        for i in range(10):
            show_state(x, y, angle)
            feature = state_to_feature(x, y, angle)
            q_value = get_q_value(feature, weight)
            max_q_value, max_action = get_max_q_action(q_value)
            action = explore_exploit(max_action, exploration_rate)
            velocity, angle_speed = action_to_physics(action)
            x, y, angle = update_state(x, y, angle, velocity, angle_speed, time_step)
            reward = get_reward(x, y, angle, velocity, time_step)
            cumulative_reward += discount * reward
            discount *= discount_factor
            logging.info('cumulative reward: %.4f', cumulative_reward)
            if state_is_end(x, y, angle):
                break
    plt.ioff()
    plt.show()


def test_q_explore_simulation():
    q_explore_simulation()


def update_weight(weight, feature, q_value, action, gt_q_value):
    learning_rate = 0.01
    weight_decay = 0.001
    delta = q_value[action] - gt_q_value
    delta_weight = delta * feature + weight_decay * weight[:, action]
    weight[:, action] = weight[:, action] - learning_rate * delta_weight
    return weight


def compute_loss(q_value, action, gt_q_value):
    loss = (q_value[action] - gt_q_value) ** 2
    # logging.info('loss: %.4f', loss)
    return loss


def q_learn():
    plt.ion()
    time_step = 1  # Discrete approximation time step for the continuous game
    train_iter = 100  # Number of training iterations
    exploration_rate = np.linspace(1.0, 0.1, train_iter)
    discount_factor = 0.9  # Discounted factor
    weight = initialize_param(3 + 1, 2)
    i = 0
    while True:
        i += 1
    #for i in range(train_iter):
        # logging.info('starting iteration %d', i)
        # x, y, angle = -4, 0, 0  # Robot state
        x = random.random() * -5
        y = random.random() * 2 - 1
        angle = random.random() * 2 * math.pi - math.pi
        cumulative_reward, discount = 0, 1
        total_loss, cnt = 0, 0
        while True:
            show_state(x, y, angle)
            feature = state_to_feature(x, y, angle)
            q_value = get_q_value(feature, weight)
            max_q_value, max_action = get_max_q_action(q_value)
            # action = explore_exploit(max_action, exploration_rate[i])
            if i >= train_iter:
                explore_rate = 0.1
            else:
                explore_rate = exploration_rate[i]
            action = explore_exploit(max_action, explore_rate)
            velocity, angle_speed = action_to_physics(action)
            x, y, angle = update_state(x, y, angle, velocity, angle_speed, time_step)
            if state_is_end(x, y, angle):
                reward = -10
                next_max_q_value = 0
            else:
                reward = get_reward(x, y, angle, velocity, time_step)
                next_feature = state_to_feature(x, y, angle)
                next_q_value = get_q_value(next_feature, weight)
                next_max_q_value, next_max_action = get_max_q_action(next_q_value)
            cumulative_reward += discount * reward
            discount *= discount_factor
            gt_q_value = reward + discount_factor * next_max_q_value
            loss = compute_loss(q_value, action, gt_q_value)
            cnt += 1
            total_loss += loss
            weight = update_weight(weight, feature, q_value, action, gt_q_value)
            if state_is_end(x, y, angle):
                break
        average_loss = total_loss / cnt
        logging.info('iteration: %d, explore rate: %.4f, cumulative reward: %.4f, average loss: %.4f', \
                     i, explore_rate, cumulative_reward, average_loss)
    plt.ioff()
    plt.show()
    return weight


def q_test(weight):
    plt.ion()
    time_step = 1  # Discrete approximation time step for the continuous game
    exploration_rate = 0.1  # Exploration rate
    discount_factor = 0.9  # Discounted factor
    test_iter = 5  # Number of training iterations
    for i in range(test_iter):
        logging.info('starting iteration %d', i)
        x, y, angle = -4, 0, 0  # Robot state
        cumulative_reward, discount = 0, 1
        while True:
            show_state(x, y, angle)
            feature = state_to_feature(x, y, angle)
            q_value = get_q_value(feature, weight)
            max_q_value, max_action = get_max_q_action(q_value)
            action = explore_exploit(max_action, exploration_rate)
            velocity, angle_speed = action_to_physics(action)
            x, y, angle = update_state(x, y, angle, velocity, angle_speed, time_step)
            if state_is_end(x, y, angle):
                reward = -10
            else:
                reward = get_reward(x, y, angle, velocity, time_step)
            cumulative_reward += discount * reward
            discount *= discount_factor
            if state_is_end(x, y, angle):
                break
        logging.info('exploration rate: %.4f, cumulative reward: %.4f', exploration_rate, cumulative_reward)
    plt.ioff()
    plt.show()


def test_functions():
    test_show_state()
    test_simple_simulation()
    test_get_reward()
    test_state_to_feature()
    test_initialize_param()
    test_get_q_value()
    test_get_max_q_action()
    test_action_to_physics()
    test_q_simulation()
    test_q_explore_simulation()


if __name__ == '__main__':
    # test_functions()
    w = q_learn()
    # w = np.array([[-0.37708407, -0.37425836],
    #              [1.27578218, 0.35215446],
    #              [-0.90236778, 0.71186919],
    #              [-1.96970796, -1.08054405]])
    print w
    q_test(w)
