# Four Rooms environment for [Gymnasium](https://gymnasium.farama.org/)

![Env Screenshot](screenshot.png)

Actions: UP, DOWN, LEFT, RIGHT
The agent and the target are randomly initialized at the start of each episode, and they are guaranteed to be in different rooms. 

# Usage

```python
import gymnasium as gym
gym.make('FourRooms-v0', render_mode="human")
```