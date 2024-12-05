from pyray import *

class Block2d:
	def __init__(
		self,
		starting_position=Vector2(100, 100),
		goal_position=Vector2(0, 0)
	):
		self.position = starting_position
		self.goal_position = goal_position
		self.direction = Vector2(0, 0)
		self.max_speed = 1200
		self.speed = Vector2(0, 0)
		self.last_update = 0 #= get_time()
		self.last_click = Vector2(-10, -10)

	def control(self, click: Vector2):
		def signum(x): return int(x > 0) - int(x < 0)
		def accelerate(v: float, extra: float) -> float:
			v = abs(v + extra) if abs(v + extra) < self.max_speed else self.max_speed
			return v
		self.last_click = click
		self.direction = Vector2(self.position.x - click.x, self.position.y - click.y)
		self.speed.x = accelerate(self.speed.x, self.direction.x)
		self.speed.y = accelerate(self.speed.y, self.direction.y)
		self.direction = Vector2(
		   signum(self.direction.x),
		   signum(self.direction.y),
		)

	def update(self):
		def decelerate(v : float) -> float:
			if v - 1.1 > 0: v -= 1.1
			else: v = 0
			return v
		update_time = self.last_update + 0.01 #get_time()
		delta_time = update_time - self.last_update
		self.position.x -= self.speed.x * delta_time * self.direction.x
		self.position.y -= self.speed.y * delta_time * self.direction.y

		self.speed.x = decelerate(self.speed.x)
		self.speed.y = decelerate(self.speed.y)

		self.last_update = update_time

	def is_win_condition(self):
		return (
			self.speed.x < 0 and
			self.speed.y < 0 and
			abs(self.position.x - self.goal_position.x) < 50 and
			abs(self.position.y - self.goal_position.y) < 50
		)
		

	def display(self):
		def draw_hitmark(x : int, y : int) -> None:
			size = 10
			draw_line(x - size, y - size, x + size, y + size, GREEN)
			draw_line(x + size, y - size, x - size, y + size, GREEN)
			draw_rectangle_lines(x - size, y - size, size*2, size*2, GREEN)
		# Blocks
		draw_rectangle(int(self.goal_position.x), int(self.goal_position.y), 30, 30, BLUE)
		draw_rectangle(int(self.position.x), int(self.position.y), 30, 30, RED)

		# Hud
		draw_text(f"X: {self.position.x:.2f}", 10, 10, 20, DARKGRAY)
		draw_text(f"Y: {self.position.y:.2f}", 10, 40, 20, DARKGRAY)
		draw_text(f"Speed X: {self.speed.x:.2f}", 10, 70, 20, DARKGRAY)
		draw_text(f"Speed Y: {self.speed.y:.2f}", 10, 100, 20, DARKGRAY)

		# Player action
		draw_hitmark(int(self.last_click.x), int(self.last_click.y))

		# Win
		if self.is_win_condition(): draw_text(f"Victory!", 200, 200, 40, DARKGRAY)
