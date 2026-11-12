"""Validate public-state MPC extension against the real MuJoCo environment."""
import numpy as np
import pytest
pytest.importorskip('gymnasium')
pytest.importorskip('mujoco')
from pgofp.envs.mujoco.planning import RolloutPlanner, double_pendulum_tip


def test_public_state_and_kinematics():
    import gymnasium as gym
    env=gym.make('InvertedDoublePendulum-v5');planner=RolloutPlanner('InvertedDoublePendulum-v5')
    try:
        for seed in range(5):
            obs,_=env.reset(seed=seed)
            state=planner.state_from_obs(obs)
            np.testing.assert_allclose(state[1:4],env.unwrapped.data.qpos,atol=1e-12)
            np.testing.assert_allclose(state[4:7],env.unwrapped.data.qvel,atol=1e-12)
            x,z=double_pendulum_tip(state[1:4]);actual=env.unwrapped.data.site_xpos[0]
            np.testing.assert_allclose([x,z],actual[[0,2]],atol=1e-12)
        # Velocity clipping must be honored rather than accessing hidden state.
        obs=obs.copy();obs[5:8]=[10,-10,10]
        np.testing.assert_array_equal(planner.state_from_obs(obs)[4:7],obs[5:8])
    finally:env.close()


def test_rollout_one_control_step_and_reward_cost():
    import gymnasium as gym
    env=gym.make('InvertedDoublePendulum-v5');planner=RolloutPlanner('InvertedDoublePendulum-v5')
    try:
        obs,_=env.reset(seed=50000);u=np.array([.003])
        qp,qv=planner.rollout(planner.state_from_obs(obs),u.reshape(1,1,1))
        _,reward,terminated,_,_=env.step(u)
        np.testing.assert_allclose(qp[0,0],env.unwrapped.data.qpos,rtol=1e-9,atol=1e-10)
        np.testing.assert_allclose(qv[0,0],env.unwrapped.data.qvel,rtol=1e-9,atol=1e-10)
        cost=planner.stage_cost(qp,qv,u.reshape(1,1,1),None,{})[0]
        np.testing.assert_allclose(cost,10-reward,atol=1e-10)
    finally:env.close()


def test_planned_fall_is_absorbing():
    planner=RolloutPlanner('InvertedDoublePendulum-v5')
    qp=np.zeros((1,3,3));qp[0,1,1]=1.0;qv=np.zeros_like(qp)
    x,z=double_pendulum_tip(qp);penalty=.01*x*x+(z-2)**2
    cost=planner.stage_cost(qp,qv,np.zeros((1,3,1)),None,{})
    np.testing.assert_allclose(cost[0],penalty.sum()+20)
