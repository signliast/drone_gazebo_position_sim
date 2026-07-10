#!/bin/bash
set -e

source /opt/ros/humble/setup.bash

if [ -f /ws/install/setup.bash ]; then
    source /ws/install/setup.bash
fi

export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-10}
export RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}

exec "$@"
