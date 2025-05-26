from environments.dmc2gym.dmc_wrappers import DMCWrapper
from gymnasium.wrappers import TimeLimit


def make(
    domain_name,
    task_name,
    resource_files,
    noise_source,
    total_frames,
    seed=1,
    visualize_reward=True,
    from_pixels=False,
    height=84,
    width=84,
    camera_id=0,
    frame_skip=1,
    episode_length=1000,
    environment_kwargs=None,
):

    if from_pixels:
        assert (
            not visualize_reward
        ), "cannot use visualize reward when learning from pixels"

    # Shorten episode length
    max_episode_steps = (episode_length + frame_skip - 1) // frame_skip

    # Instantiate the environment directly
    env = DMCWrapper(
        domain_name=domain_name,
        task_name=task_name,
        resource_files=resource_files,
        noise_source=noise_source,
        total_frames=total_frames,
        task_kwargs={"random": seed},
        environment_kwargs=environment_kwargs,
        visualize_reward=visualize_reward,
        from_pixels=from_pixels,
        height=height,
        width=width,
        camera_id=camera_id,
        frame_skip=frame_skip,
    )

    # Wrap the environment with a time limit (equivalent to max_episode_steps)
    env = TimeLimit(env, max_episode_steps=max_episode_steps)

    return env
