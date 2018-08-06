import numpy as np
import tensorflow as tf
import gym
import logz
import scipy.signal
import os
import time
import inspect
from multiprocessing import Process


def build_mlp(
        input_placeholder,
        output_size,
        scope,
        n_layers=2,
        size=64,
        activation=tf.tanh,
        output_activation=None
        ):

    with tf.variable_scope(scope):
        # YOUR_CODE_HERE
        hidden_layer = input_placeholder
        for i in range(n_layers):
            hidden_layer = tf.layers.dense(hidden_layer, size, activation=activation)
        return tf.layers.dense(hidden_layer, output_size, activation=output_activation)


def pathlength(path):
    return len(path["reward"])


#============================================================================================#
# Policy Gradient
#============================================================================================#


def train_PG(exp_name='',
             env_name='CartPole-v0',
             n_iter=100,
             gae_lambda=1.0,
             gamma=1.0,
             min_timesteps_per_batch=1000,
             max_path_length=None,
             learning_rate=5e-3,
             reward_to_go=True,
             animate=True,
             logdir=None,
             normalize_advantages=True,
             nn_baseline=False,
             seed=0,
             # network arguments
             n_layers=1,
             size=32
             ):

    start = time.time()

    # Configure output directory for logging
    logz.configure_output_dir(logdir)

    # Log experimental parameters
    args = inspect.getargspec(train_PG)[0]
    locals_ = locals()
    params = {k: locals_[k] if k in locals_ else None for k in args}
    logz.save_params(params)

    # Set random seeds
    tf.set_random_seed(seed)
    np.random.seed(seed)

    # Make the gym environment
    env = gym.make(env_name)

    # Is this env continuous, or discrete?
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    # Maximum length for episodes
    max_path_length = max_path_length or env.spec.max_episode_steps

    # Observation and action sizes
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

    #========================================================================================#
    #                           ----------SECTION 4----------
    # Placeholders
    #
    # Need these for batch observations / actions / advantages in policy gradient loss function.
    #========================================================================================#

    sy_ob_no = tf.placeholder(shape=[None, ob_dim], name="ob", dtype=tf.float32)
    if discrete:
        sy_ac_na = tf.placeholder(shape=[None], name="ac", dtype=tf.int32)
    else:
        sy_ac_na = tf.placeholder(shape=[None, ac_dim], name="ac", dtype=tf.float32)

    # Define a placeholder for advantages
    sy_adv_n = tf.placeholder(shape=[None], name='adv', dtype=tf.float32)

    if discrete:
        # YOUR_CODE_HERE
        sy_logits_na = build_mlp(input_placeholder=sy_ob_no,
                                 output_size=ac_dim,
                                 scope='discrete',
                                 n_layers=n_layers,
                                 size=size)
        sy_sampled_ac = tf.reshape(tf.multinomial(sy_logits_na, 1), [-1]) # Hint: Use the tf.multinomial op
        sy_logprob_n = -tf.nn.sparse_softmax_cross_entropy_with_logits(labels=sy_ac_na, logits=sy_logits_na)

    else:
        # YOUR_CODE_HERE
        sy_mean = build_mlp(input_placeholder=sy_ob_no,
                            output_size=ac_dim,
                            scope='continus',
                            n_layers=n_layers,
                            size=size)
        sy_logstd = tf.get_variable(name='logstd', shape=[ac_dim], dtype=tf.float32) # logstd should just be a trainable variable, not a network output.
        sy_sampled_ac = tf.random_normal(shape=tf.shape(sy_mean), mean=sy_mean, stddev=tf.exp(sy_logstd))
        dist = tf.contrib.distributions.MultivariateNormalDiag(loc=sy_mean, scale=tf.exp(sy_logstd))
        sy_logprob_n = dist.log_prob(sy_ac_na)
    # Hint: Use the log probability under a multivariate gaussian.

    #========================================================================================#
    #                           ----------SECTION 4----------
    # Loss Function and Training Operation
    #========================================================================================#

    loss = tf.reduce_mean(-sy_logprob_n*sy_adv_n) # Loss function that we'll differentiate to get the policy gradient.
    update_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)


    #========================================================================================#
    #                           ----------SECTION 5----------
    # Optional Baseline
    #========================================================================================#

    if nn_baseline:
        baseline_prediction = tf.squeeze(build_mlp(
                                sy_ob_no,
                                1,
                                "nn_baseline",
                                n_layers=n_layers,
                                size=size))
        # Define placeholders for targets, a loss function and an update op for fitting a
        # neural network baseline. These will be used to fit the neural network baseline.
        # YOUR_CODE_HERE
        baseline_targets = tf.placeholder(shape=[None], name='targets', dtype=tf.float32)
        baseline_loss = tf.nn.l2_loss(baseline_prediction - baseline_targets)
        baseline_update_op = tf.train.AdamOptimizer(learning_rate).minimize(baseline_loss)


    #========================================================================================#
    # Tensorflow Engineering: Config, Session, Variable initialization
    #========================================================================================#
    tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1,
                               allow_soft_placement=True,
                               log_device_placement=False
                               )

    sess = tf.Session(config=tf_config)
    sess.__enter__() # equivalent to `with sess:`
    tf.global_variables_initializer().run() #pylint: disable=E1101

    # ========================================================================================#
    # Training Loop
    # ========================================================================================#

    total_timesteps = 0

    for itr in range(n_iter):
        print("********** Iteration %i ************"%itr)

        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        paths = []
        while True:
            ob = env.reset()
            obs, acs, rewards = [], [], []
            animate_this_episode=(len(paths)==0 and (itr % 10 == 0) and animate)
            steps = 0
            while True:
                if animate_this_episode:
                    env.render()
                    time.sleep(0.05)
                obs.append(ob)
                ac = sess.run(sy_sampled_ac, feed_dict={sy_ob_no : ob[None]})
                ac = ac[0]
                acs.append(ac)
                ob, rew, done, _ = env.step(ac)
                rewards.append(rew)
                steps += 1
                if done or steps > max_path_length:
                    break
            path = {"observation" : np.array(obs),
                    "reward" : np.array(rewards),
                    "action" : np.array(acs)}
            paths.append(path)
            timesteps_this_batch += pathlength(path)
            if timesteps_this_batch > min_timesteps_per_batch:
                break
        total_timesteps += timesteps_this_batch

        # Build arrays for observation, action for the policy gradient update by concatenating
        # across paths
        ob_no = np.concatenate([path["observation"] for path in paths])
        ac_na = np.concatenate([path["action"] for path in paths])

        # YOUR_CODE_HERE
        q_n = []
        for path in paths:
            r = path['reward']
            max_step = len(r)
            q = np.zeros(max_step)
            q[-1] = r[-1]
            for t in reversed(range(max_step - 1)):
                q[t] = r[t] + gamma * q[t + 1]
            q_n.extend(q)
            if not reward_to_go:
                q_n.extend([q[0]] * max_step)
        q_n = np.array(q_n)
        #====================================================================================#
        #                           ----------SECTION 5----------
        # Computing Baselines
        #====================================================================================#

        if nn_baseline:
            b_n = sess.run(baseline_prediction, feed_dict={sy_ob_no: ob_no})
            # b_n = b_n * np.std(q_n, axis=0) + np.mean(q_n, axis=0)
            # adv_n = q_n - b_n
            adv_n = []
            idx = 0
            for path in paths:
                r = path['reward']
                max_step = len(r)
                adv = np.zeros(max_step)
                adv[-1] = r[-1]
                for t in reversed(range(max_step - 1)):
                    delta = r[t] + b_n[idx + t + 1] - b_n[idx + t]
                    adv[t] = delta + gae_lambda * gamma * adv[t + 1]
                idx += max_step
                adv_n.extend(adv)
            q_n = b_n + adv_n
        else:
            adv_n = q_n.copy()

        #====================================================================================#
        #                           ----------SECTION 4----------
        # Advantage Normalization
        #====================================================================================#

        if normalize_advantages:
            # On the next line, implement a trick which is known empirically to reduce variance
            # in policy gradient methods: normalize adv_n to have mean zero and std=1.
            # YOUR_CODE_HERE
            mean_adv = np.mean(adv_n, axis=0)
            std_adv = np.std(adv_n, axis=0)
            adv_n = (adv_n - mean_adv) / (std_adv + 1e-7)
            pass


        #====================================================================================#
        #                           ----------SECTION 5----------
        # Optimizing Neural Network Baseline
        #====================================================================================#
        if nn_baseline:
            q_n_mean = np.mean(q_n)
            q_n_std = np.std(q_n)
            q_n = (q_n - q_n_mean) / (q_n_std + 1e-7)
            sess.run(baseline_update_op, feed_dict={sy_ob_no: ob_no, baseline_targets: q_n})
            pass

        #====================================================================================#
        #                           ----------SECTION 4----------
        # Performing the Policy Update
        #====================================================================================#

        # Call the update operation necessary to perform the policy gradient update based on
        # the current batch of rollouts.
        #
        # For debug purposes, you may wish to save the value of the loss function before
        # and after an update, and then log them below.

        # YOUR_CODE_HERE
        sess.run(update_op, feed_dict={sy_ob_no: ob_no, sy_ac_na: ac_na, sy_adv_n: adv_n})

        # Log diagnostics
        returns = [path["reward"].sum() for path in paths]
        ep_lengths = [pathlength(path) for path in paths]
        logz.log_tabular("Time", time.time() - start)
        logz.log_tabular("Iteration", itr)
        logz.log_tabular("AverageReturn", np.mean(returns))
        logz.log_tabular("StdReturn", np.std(returns))
        logz.log_tabular("MaxReturn", np.max(returns))
        logz.log_tabular("MinReturn", np.min(returns))
        logz.log_tabular("EpLenMean", np.mean(ep_lengths))
        logz.log_tabular("EpLenStd", np.std(ep_lengths))
        logz.log_tabular("TimestepsThisBatch", timesteps_this_batch)
        logz.log_tabular("TimestepsSoFar", total_timesteps)
        logz.dump_tabular()
        logz.pickle_tf_vars()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str)
    parser.add_argument('--exp_name', type=str, default='vpg')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--gae_lambda', '-gae_lambda', type=float, default=0.8)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--n_iter', '-n', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=1000)
    parser.add_argument('--ep_len', '-ep', type=float, default=-1.)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--reward_to_go', '-rtg', action='store_true')
    parser.add_argument('--dont_normalize_advantages', '-dna', action='store_true')
    parser.add_argument('--nn_baseline', '-bl', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_experiments', '-e', type=int, default=1)
    parser.add_argument('--n_layers', '-l', type=int, default=1)
    parser.add_argument('--size', '-s', type=int, default=32)
    args = parser.parse_args()

    if not(os.path.exists('data')):
        os.makedirs('data')
    logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('data', logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    max_path_length = args.ep_len if args.ep_len > 0 else None

    for e in range(args.n_experiments):
        seed = args.seed + 10*e
        print('Running experiment with seed %d'%seed)
        def train_func():
            train_PG(
                exp_name=args.exp_name,
                env_name=args.env_name,
                n_iter=args.n_iter,
                gae_lambda=args.gae_lambda,
                gamma=args.discount,
                min_timesteps_per_batch=args.batch_size,
                max_path_length=max_path_length,
                learning_rate=args.learning_rate,
                reward_to_go=args.reward_to_go,
                animate=args.render,
                logdir=os.path.join(logdir,'%d'%seed),
                normalize_advantages=not(args.dont_normalize_advantages),
                nn_baseline=args.nn_baseline,
                seed=seed,
                n_layers=args.n_layers,
                size=args.size
                )
        # Awkward hacky process runs, because Tensorflow does not like
        # repeatedly calling train_PG in the same thread.
        p = Process(target=train_func, args=tuple())
        p.start()
        p.join()


if __name__ == "__main__":
    main()
