import glob

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


def get_section_results(file):
    """
        requires tensorflow==1.12.0
    """
    X = []
    Y = []
    for e in tf.compat.v1.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'Train_EnvstepsSoFar':
                X.append(v.simple_value)
            elif v.tag == 'Train_BestReturn':
                Y.append(v.simple_value)
    return X, Y

if __name__ == '__main__':
    import glob

    logdir = 'D:/pryor/Documents/GitHubProjects/DeepRL-Course/hw1/data/q1_*/events*'
    files = glob.glob(logdir)

    runs = []
    for eventfile in files:
        X, Y = get_section_results(eventfile)
        runs.append((X[:286], Y[:286]))

    # Plot individual runs with fading
    plt.figure(figsize=(10, 6))

    for i, (X, Y) in enumerate(runs):
        plt.plot(X, Y, label=f'Run {i + 1}', alpha=0.5)

    # Plot the average line
    average_X = np.mean([X for X, _ in runs], axis=0)
    average_Y = np.mean([Y for _, Y in runs], axis=0)
    plt.plot(average_X, average_Y, label='Average', linewidth=2, color='black')
    plt.legend()
    plt.title("DQN on Ms Pacman")
    plt.ylabel("Best Return")
    plt.xlabel("Env Steps")
    plt.show()

    logdir = 'D:/pryor/Documents/GitHubProjects/DeepRL-Course/hw1/data/q2_doubledqn_*/events*'
    files = glob.glob(logdir)

    # runs_double = []
    # for eventfile in files:
    #     X, Y = get_section_results(eventfile)
    #     runs_double.append((X[1:], Y))
    #
    # # Plot individual runs with fading
    # plt.figure(figsize=(10, 6))
    #
    # for i, (X, Y) in enumerate(runs_double):
    #     plt.plot(X, Y, label=f'Run {i + 1}', alpha=0.5)
    #
    # # Plot the average line
    # average_X = np.mean([X for X, _ in runs_double], axis=0)
    # average_Y = np.mean([Y for _, Y in runs_double], axis=0)
    # plt.plot(average_X, average_Y, label='Average', linewidth=2, color='black')
    # plt.legend()
    # plt.title("Double DQN")
    # plt.ylabel("Average Return")
    # plt.xlabel("Env Steps")
    # plt.show()
    #
    #
    # # Plot individual runs with fading
    # green_shades = [(0, 128, 0), (0, 160, 0), (0, 192, 0)]
    # red_shades = [(255, 0, 0), (220, 20, 60), (178, 34, 34)]
    # green_shades = [(r / 255, g / 255, b / 255) for r, g, b in green_shades]
    # red_shades = [(r / 255, g / 255, b / 255) for r, g, b in red_shades]
    #
    # plt.figure(figsize=(10, 6))
    #
    # for i, (X, Y) in enumerate(runs_double):
    #     plt.plot(X, Y, label=f'DDQN {i + 1}', alpha=0.25, color=green_shades[i])
    # for i, (X, Y) in enumerate(runs):
    #     plt.plot(X, Y, label=f'DQN {i + 1}', alpha=0.25, linestyle='dashed', color=red_shades[i])
    #
    # # Plot the average line
    # average_X = np.mean([X for X, _ in runs_double], axis=0)
    # average_Y = np.mean([Y for _, Y in runs_double], axis=0)
    # plt.plot(average_X, average_Y, label='Average DDQN', linewidth=2, color='green')
    #
    # # Plot the average line
    # average_X = np.mean([X for X, _ in runs], axis=0)
    # average_Y = np.mean([Y for _, Y in runs], axis=0)
    # plt.plot(average_X, average_Y, label='Average DQN', linewidth=2, color='red')
    #
    # plt.legend()
    # plt.title("DDQN vs DQN")
    # plt.ylabel("Average Return")
    # plt.xlabel("Env Steps")
    # plt.show()