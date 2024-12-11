clean:
	-rm -r logs/PPO_* logs/A2C_*
	-rm frame_*.png

gif:
	convert -delay 10 -loop 0 *.png output.gif
