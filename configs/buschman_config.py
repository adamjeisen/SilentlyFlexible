def get_config():
    config = dict(
        type='buschman',
        simulation_kwargs=dict(
            load=7,
            N_rand=2560
        )
    )
    return config
