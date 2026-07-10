FROM ros:humble-ros-base AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=humble

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-colcon-common-extensions \
    python3-opencv \
    ros-humble-rclpy \
    ros-humble-cv-bridge \
    ros-humble-std-msgs \
    ros-humble-sensor-msgs \
    ros-humble-geometry-msgs \
    ros-humble-image-transport \
    ros-humble-ros-gz-bridge \
    ros-humble-mavros \
    ros-humble-mavros-extras \
    git \
    nano \
    vim \
    curl \
    wget \
    net-tools \
    iputils-ping \
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /ws

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]


FROM base AS dev

RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-ros-gz-sim \
    ros-humble-rqt-image-view \
    x11-apps \
    && rm -rf /var/lib/apt/lists/*


FROM base AS pi

CMD ["bash"]
