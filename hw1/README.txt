
1) See hw1 if you'd like to see installation instructions. You do NOT have to redo them.


##############################################
##############################################


2) Code:

-------------------------------------------

Files to look at, even though there are no explicit 'TODO' markings:
- scripts/run_hw2_policy_gradient.py

-------------------------------------------

Blanks to be filled in by using your code from hw1 are marked with 'TODO: GETTHIS from HW1'

The following files have these:
- infrastructure/rl_trainer.py
- infrastructure/utils.py
- policies/MLP_policy.py

-------------------------------------------

Blanks to be filled in now (for this assignment) are marked with 'TODO'

The following files have these:
- agents/pg_agent.py
- policies/MLP_policy.py


##############################################
##############################################


3) Run code with the following command:

$ python ds6559/scripts/run_hw2.py --env_name CartPole-v0 --exp_name test_cartpole --video_log_freq -1
$ python ds6559/scripts/run_hw2.py --env_name InvertedPendulum-v2 --exp_name test_pendulum --video_log_freq -1

To speed up your runs, remove --video_log_freq -1 (so, you will have no video saved).

Flags of relevance, when running the commands above (see pdf for more info):
-n number of policy training iterations
-rtg use reward_to_go for the value
-dsa do not standardize the advantage values

##############################################


4) Visualize saved tensorboard event file:

$ cd ds6559/data/<your_log_dir>
$ tensorboard --logdir .

Then, navigate to shown url to see scalar summaries as plots (in 'scalar' tab), as well as videos (in 'images' tab)

