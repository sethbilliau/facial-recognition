xhost +local:root

sudo docker run --runtime nvidia -it --rm --network host \
   --volume ~/Documents/GitHub/bah-facial-recognition:/data \
   --volume /tmp/.X11-unix:/tmp/.X11-unix \
   --device /dev/video0 \
   -e DISPLAY=$DISPLAY \
   face:v2
