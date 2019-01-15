#!/usr/bin/env python3
from baselines.common.cmd_util import make_mujoco_env, mujoco_arg_parser
from baselines.common import tf_util as U
from baselines import logger
from distilation.config import *
import tensorflow as tf
from baselines.ppo1 import mlp_policy, pposgd_simple
######################## AGENTS ###############################
#                                                             #
###############################################################

class TeacherAgent(object):
    def __init__(self,env,sess,restore, batch=1):
      self.pi = mlp_policy.MlpPolicy(name='pi',
              ob_space=env.observation_space, ac_space=env.action_space,
              hid_size=64, num_hid_layers=2, training_batch_size=batch)
      self.saver = tf.train.Saver(
          var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pi'))
      if restore:
          self.saver.restore(sess, "{0}/teacher.ckpt".format(base_path))


def train(env_id, num_timesteps, seed):
    from baselines.ppo1 import mlp_policy, pposgd_simple
    U.make_session(num_cpu=1).__enter__()
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)
    env = make_mujoco_env(env_id, seed)
    pposgd_simple.learn(env, policy_fn,
            max_timesteps=num_timesteps,
            timesteps_per_actorbatch=2048,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
            gamma=0.99, lam=0.95, schedule='linear',
        )
    env.close()

def collect_reward():
    env = make_mujoco_env("Reacher-v2", 0)
    sess = tf.Session()
    teacher = TeaherAgent(env,sess,restore=True)
    dataset = Dataset(dir_path="/home/winstonww/reacher/data/dataset_teacher/")
    ob = env.reset()
    while True:
        # Get Teacher action for the last observation
        t_mean  = sess.run(
            ( teacher.pi.pd.mean ), 
            feed_dict={ ob_ph: np.expand_dims( ob, axis=0 ) } )


        dataset.write(
            ob=ob,
            reward=reward,
            t_pdflat=t_pdflat,
            s_pdflat=np.zeros([PDFLAT_SHAPE]),
            stepped_with='t')

        ob, reward, new, _ = env.step( t_mean )


        if new:
            ob = env.reset()
            dataset.flush()

def main():
    args = mujoco_arg_parser().parse_args()
    logger.configure()
    #train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)

if __name__ == '__main__':
    main()
