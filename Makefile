clean:
	-rm -r logs/PPO_*
	-rm frame_*.png

gif:
	convert -delay 10 -loop 0 *.png output.gif
