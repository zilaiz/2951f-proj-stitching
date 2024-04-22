from rlkit.torch.networks import ConcatMlp


def get_gciql_models(obs_dim, action_dim, goal_dim, qf_kwargs, vf_kwargs):
    # qf_kwargs = variant.get("qf_kwargs", {})
    # goal_dim = obs_dim

    qf1 = ConcatMlp(
        input_size=obs_dim + action_dim + goal_dim,
        output_size=1,
        **qf_kwargs
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + action_dim + goal_dim,
        output_size=1,
        **qf_kwargs
    )
    target_qf1 = ConcatMlp(
        input_size=obs_dim + action_dim + goal_dim,
        output_size=1,
        **qf_kwargs
    )
    target_qf2 = ConcatMlp(
        input_size=obs_dim + action_dim + goal_dim,
        output_size=1,
        **qf_kwargs
    )

    # vf_kwargs = variant.get("vf_kwargs", dict(hidden_sizes=[256, 256, ],))
    
    vf = ConcatMlp(
        input_size=obs_dim + goal_dim,
        output_size=1,
        **vf_kwargs
    )

    return qf1, qf2, target_qf1, target_qf2, vf

def get_gciql_policy():
    pass