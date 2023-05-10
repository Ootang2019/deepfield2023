#!/bin/bash

# stopping all rosnodes
rosnode kill --all
killall -9 roslaunch
killall -9 roslaunch
killall -9 roslaunch
killall rosmaster

pkill -9 -f rl.py


