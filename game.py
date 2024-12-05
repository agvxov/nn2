from pyray import *
from block2d import Block2d

screen_width = 800
screen_height = 600

def main():
	init_window(screen_width, screen_height, "Physics Simulation")
	set_target_fps(60)

	content = Block2d(
		Vector2(screen_width / 2, screen_height / 2),
		Vector2(screen_width / 3, screen_height / 3),
	)

	while not window_should_close():
		if is_mouse_button_pressed(MOUSE_LEFT_BUTTON):
			content.control(get_mouse_position())

		content.update()

		begin_drawing()
		clear_background(RAYWHITE)
		content.display()
		end_drawing()

	close_window()

if __name__ == "__main__":
	main()
