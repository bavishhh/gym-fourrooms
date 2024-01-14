from gymnasium.envs.registration import register

register(
  id="FourRooms-v0",
  entry_point="gridworld.fourrooms:FourRoomsEnv"
)