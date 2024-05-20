from gymnasium.envs.registration import register

register(
    id='GridWorld-v0',
    entry_point='environments:GridWorldEnv',
)

register(
    id='TaxiEnv-v3',
    entry_point='environments:TaxiEnv',
)

register(
    id="Watermaze-v0",
    entry_point='environments:Watermaze',
)

register(
    id="TicTacToe-v0",
    entry_point='environments:TicTacToe',
)

register(
    id="TwoStateMDP-v0",
    entry_point='environments:TwoStateMDP',
)

register(
    id="ThreeStateMDP-v0",
    entry_point='environments:ThreeStateMDP',
)
