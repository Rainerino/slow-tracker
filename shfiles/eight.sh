sudo chmod 777 /dev/ttyACM0 & sleep 2;
roslaunch realsense2_camera rs_camera.launch & sleep 10;
roslaunch mavros px4.launch & sleep 10;
rosrun mavros mavcmd long 511 105 2500 0 0 0 0 0 & sleep 2;
rosrun mavros mavcmd long 511 31 2500 0 0 0 0 0 & sleep 2;
roslaunch vins fast_drone_250.launch & sleep 2;
roslaunch min_snap real.launch & sleep 2;
roslaunch px4ctrl run_ctrl.launch & sleep 2;
wait;